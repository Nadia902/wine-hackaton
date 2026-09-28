FROM python:3.12-slim

# System dependencies for OpenCV + EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache: install dependencies — CPU-only PyTorch first (much smaller)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models
RUN python -c "import easyocr; easyocr.Reader(['ru', 'en'], gpu=False, download_enabled=True)"

# Application code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY data/ /app/data/
COPY models/ /app/models/
COPY opt_dataset/ /app/opt_dataset/

EXPOSE 8000

CMD ["python", "-m", "backend.app"]
