import os
import cv2
from celery import Celery
from upscaler import RealESRGANUpscaler

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("acuity_worker", broker=REDIS_URL, backend=REDIS_URL)

# Global upscaler instances loaded once per worker process
upscalers = {}

def get_upscaler(model_name="realesrgan"):
    if model_name not in upscalers:
        if model_name == "realesrgan":
            up = RealESRGANUpscaler()
            up.load()
            upscalers[model_name] = up
        # SwinIR will be added in step 5
        else:
            raise ValueError(f"Unknown model {model_name}")
    return upscalers[model_name]

@celery_app.task(name="upscale_image")
def upscale_image_task(job_id: str, input_path: str, output_path: str, model_name: str, scale: int, face_enhance: bool):
    try:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to load image")
            
        upscaler = get_upscaler(model_name)
        out_img = upscaler.upscale(img, scale=scale)
        
        cv2.imwrite(output_path, out_img)
        return {"status": "done", "result_url": f"/api/download/{job_id}"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
