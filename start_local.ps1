# This script sets up and runs the backend locally without Docker or Redis

Write-Host "Setting up local backend..." -ForegroundColor Cyan

# 1. Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    . "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    . "venv\Scripts\Activate.ps1"
}

# 2. Upgrade pip and install build essentials
Write-Host "Installing dependencies... (this may take a while as it downloads PyTorch)" -ForegroundColor Yellow
pip install --upgrade pip
pip install setuptools wheel numpy Cython

# 3. Install basicsr with no build isolation to fix setuptools errors
pip install basicsr --no-build-isolation

# 4. Install PyTorch and other requirements
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r backend/requirements.txt

# 5. Start the backend in SYNC mode (bypasses Celery/Redis)
Write-Host "Starting API Server..." -ForegroundColor Green
$env:SYNC_MODE = "1"
cd backend
uvicorn main:app --port 8000 --reload
