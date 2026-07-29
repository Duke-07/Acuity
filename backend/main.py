import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

# from upscaler import RealESRGANUpscaler

app = FastAPI(title="Acuity Upscaler API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# upscaler = RealESRGANUpscaler()
# upscaler.load()

jobs_db = {}

@app.post("/api/upscale")
async def upscale(
    file: UploadFile,
    model: str = Form("realesrgan"),
    scale: int = Form(4),
    face_enhance: bool = Form(False)
):
    if not file.content_type in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    job_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}_{file.filename}")

    with open(input_path, "wb") as f:
        f.write(file_bytes)

    # Validate image dimensions
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    if img.shape[0] > 4000 or img.shape[1] > 4000:
        raise HTTPException(status_code=400, detail="Image too large (max 4000x4000)")

    # For step 3: synchronous processing
    jobs_db[job_id] = {"status": "processing"}
    
    try:
        # out_img = upscaler.upscale(img, scale=scale)
        # cv2.imwrite(output_path, out_img)
        jobs_db[job_id] = {"status": "done", "result_url": f"/api/download/{job_id}"}
    except Exception as e:
        jobs_db[job_id] = {"status": "failed", "error": str(e)}

    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs_db[job_id]

@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs_db[job_id]["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not finished")
    
    # Find the output file
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith(job_id):
            return FileResponse(os.path.join(OUTPUT_DIR, f))
    
    raise HTTPException(status_code=404, detail="File not found")
