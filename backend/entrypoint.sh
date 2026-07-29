#!/bin/bash
set -e

echo "Starting container..."

# The models directory is mounted to /models by docker-compose
MODEL_DIR=${MODEL_DIR:-/models}
mkdir -p "$MODEL_DIR"

# URLs
REALESRGAN_X4PLUS="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
REALESRGAN_ANIME="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
SWINIR="https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-L_x4_GAN.pth"
GFPGAN="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"

# Download models if they don't exist
if [ ! -f "$MODEL_DIR/RealESRGAN_x4plus.pth" ]; then
    echo "Downloading RealESRGAN_x4plus.pth..."
    curl -L -o "$MODEL_DIR/RealESRGAN_x4plus.pth" $REALESRGAN_X4PLUS
fi

if [ ! -f "$MODEL_DIR/RealESRGAN_x4plus_anime_6B.pth" ]; then
    echo "Downloading RealESRGAN_x4plus_anime_6B.pth..."
    curl -L -o "$MODEL_DIR/RealESRGAN_x4plus_anime_6B.pth" $REALESRGAN_ANIME
fi

if [ ! -f "$MODEL_DIR/003_realSR_BSRGAN_DFO_s64w8_SwinIR-L_x4_GAN.pth" ]; then
    echo "Downloading SwinIR..."
    curl -L -o "$MODEL_DIR/003_realSR_BSRGAN_DFO_s64w8_SwinIR-L_x4_GAN.pth" $SWINIR
fi

if [ ! -f "$MODEL_DIR/GFPGANv1.3.pth" ]; then
    echo "Downloading GFPGANv1.3.pth..."
    curl -L -o "$MODEL_DIR/GFPGANv1.3.pth" $GFPGAN
fi

echo "All models present."

# Run the CMD
exec "$@"
