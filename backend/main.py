import os
import uuid
from fastapi import FastAPI, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import os
import uuid
from worker import celery_app, upscale_image_task
from celery.result import AsyncResult

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

# For local testing without Redis
LOCAL_JOBS = {}

@app.post("/api/upscale")
async def upscale(
    file: UploadFile,
    background_tasks: BackgroundTasks,
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

    if os.environ.get("SYNC_MODE") == "1":
        LOCAL_JOBS[job_id] = "processing"
        
        def run_sync():
            try:
                upscale_image_task(job_id, input_path, output_path, model, scale, face_enhance)
                LOCAL_JOBS[job_id] = "done"
            except Exception as e:
                print(f"Error in sync mode: {e}")
                LOCAL_JOBS[job_id] = "failed"
                
        background_tasks.add_task(run_sync)
    else:
        task = upscale_image_task.apply_async(
            args=[job_id, input_path, output_path, model, scale, face_enhance],
            task_id=job_id
        )

    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if os.environ.get("SYNC_MODE") == "1":
        state = LOCAL_JOBS.get(job_id, "unknown")
        if state == "done":
            return {"status": "done", "result_url": f"/api/download/{job_id}"}
        return {"status": state}

    res = AsyncResult(job_id, app=celery_app)
    if res.state == 'PENDING':
        return {"status": "queued"}
    elif res.state == 'STARTED' or res.state == 'RETRY':
        return {"status": "processing"}
    elif res.state == 'SUCCESS':
        return res.result
    elif res.state == 'FAILURE':
        return {"status": "failed", "error": str(res.info)}
    else:
        return {"status": "unknown"}

@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    res = AsyncResult(job_id, app=celery_app)
    if res.state != 'SUCCESS':
        raise HTTPException(status_code=400, detail="Job not finished")
    
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith(job_id):
            return FileResponse(os.path.join(OUTPUT_DIR, f))
    
    raise HTTPException(status_code=404, detail="File not found")
