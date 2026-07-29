import os
import requests
import numpy as np
from abc import ABC, abstractmethod
import torch

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
        self.scale = 4  # RealESRGAN_x4plus is x4 by default

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

        # tile=400 on CPU to avoid using too much memory and making it slow/crash
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
