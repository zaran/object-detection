FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies (OpenMP for torch, and cleanup)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Pre-copy requirements to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install CPU-only torch + other deps
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio && \
    pip install --no-cache-dir -r requirements.txt

# Copy rest of the source code
COPY . .

# Expose port for app
EXPOSE 8000

# Start the app using gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
