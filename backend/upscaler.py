import os
import requests
import numpy as np
import math
from abc import ABC, abstractmethod
import torch
import cv2

class Upscaler(ABC):
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def upscale(self, image: np.ndarray, scale: int) -> np.ndarray:
        pass

def download_model(url: str, dest_path: str):
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"Downloading {url} to {dest_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

class RealESRGANUpscaler(Upscaler):
    def __init__(self, model_name="RealESRGAN_x4plus", model_dir="models"):
        self.model_name = model_name
        self.model_dir = model_dir
        self.upsampler = None
        self.scale = 4

    def load(self) -> None:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        model_path = os.path.join(self.model_dir, f"{self.model_name}.pth")
        
        if self.model_name == "RealESRGAN_x4plus":
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            model_cls = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        elif self.model_name == "RealESRGAN_x4plus_anime_6B":
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            model_cls = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        download_model(url, model_path)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        half = True if torch.cuda.is_available() else False

        self.upsampler = RealESRGANer(
            scale=self.scale,
            model_path=model_path,
            dni_weight=None,
            model=model_cls,
            tile=400 if not torch.cuda.is_available() else 0,
            tile_pad=10,
            pre_pad=0,
            half=half,
            device=device
        )
        print(f"RealESRGAN loaded on {device}")

    def upscale(self, image: np.ndarray, scale: int) -> np.ndarray:
        if self.upsampler is None:
            self.load()
        
        output, _ = self.upsampler.enhance(image, outscale=scale)
        return output

class SwinIRUpscaler(Upscaler):
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.model = None
        self.device = None
        self.scale = 4

    def load(self) -> None:
        from network_swinir import SwinIR
        model_path = os.path.join(self.model_dir, "003_realSR_BSRGAN_DFO_s64w8_SwinIR-L_x4_GAN.pth")
        url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-L_x4_GAN.pth"
        
        download_model(url, model_path)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # SwinIR-L parameters
        self.model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                       num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                       mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        
        pretrained_dict = torch.load(model_path, map_location=self.device)
        param_key_g = 'params'
        self.model.load_state_dict(pretrained_dict[param_key_g] if param_key_g in pretrained_dict.keys() else pretrained_dict, strict=True)
        
        self.model.eval()
        self.model = self.model.to(self.device)
        print(f"SwinIR loaded on {self.device}")

    def upscale(self, image: np.ndarray, scale: int) -> np.ndarray:
        if self.model is None:
            self.load()

        # OpenCV is BGR, we need RGB for SwinIR, range [0, 1]
        img_lq = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_lq = img_lq.astype(np.float32) / 255.
        
        # HWC to CHW
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [0, 1, 2]], (2, 0, 1))
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)

        # Tiled inference
        window_size = 8
        tile = 400
        tile_overlap = 32

        b, c, h, w = img_lq.shape
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile] if h > tile else [0]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile] if w > tile else [0]
        
        E = torch.zeros(b, c, h*self.scale, w*self.scale).type_as(img_lq)
        W = torch.zeros_like(E)
        
        with torch.no_grad():
            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    
                    # Pad to multiple of window_size
                    _, _, h_in, w_in = in_patch.shape
                    pad_h = (window_size - h_in % window_size) % window_size
                    pad_w = (window_size - w_in % window_size) % window_size
                    in_patch = torch.nn.functional.pad(in_patch, (0, pad_w, 0, pad_h), 'reflect')
                    
                    out_patch = self.model(in_patch)
                    
                    # Crop back if padded
                    out_patch = out_patch[..., :h_in*self.scale, :w_in*self.scale]
                    
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx*self.scale:(h_idx+tile)*self.scale, w_idx*self.scale:(w_idx+tile)*self.scale].add_(out_patch)
                    W[..., h_idx*self.scale:(h_idx+tile)*self.scale, w_idx*self.scale:(w_idx+tile)*self.scale].add_(out_patch_mask)
        
        output = E.div_(W)
        
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[0, 1, 2], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        
        # RGB to BGR
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # If requested scale is different from model scale (4), resize
        if scale != self.scale:
            h_out, w_out = output.shape[:2]
            output = cv2.resize(output, (int(w_out * scale / self.scale), int(h_out * scale / self.scale)), interpolation=cv2.INTER_LANCZOS4)
            
        return output
