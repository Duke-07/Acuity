import cv2
import sys
import os
import argparse
from upscaler import RealESRGANUpscaler

# Monkeypatch torchvision for basicsr compatibility with newer PyTorch
try:
    import torchvision
    from torchvision.transforms import functional_tensor
except ImportError:
    import torchvision
    import torchvision.transforms.functional as functional_tensor
    import sys
    sys.modules['torchvision.transforms.functional_tensor'] = functional_tensor

def main():
    parser = argparse.ArgumentParser(description="Test Real-ESRGAN upscaling")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output image path")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file {args.input} does not exist.")
        sys.exit(1)

    print(f"Loading image {args.input}...")
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    
    print("Initializing Real-ESRGAN...")
    upscaler = RealESRGANUpscaler()
    upscaler.load()

    print("Upscaling...")
    out_img = upscaler.upscale(img, scale=args.scale)
    
    print(f"Saving to {args.output}...")
    cv2.imwrite(args.output, out_img)
    print("Done!")

if __name__ == "__main__":
    main()
