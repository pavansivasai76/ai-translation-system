# =========================
# Base Image
# =========================
FROM python:3.11-slim

# =========================
# Prevent Python buffering
# =========================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# =========================
# Install System Dependencies
# =========================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Set Work Directory
# =========================
WORKDIR /app

# =========================
# Copy Requirements First (Docker Cache Optimization)
# =========================
COPY Backend/requirements.txt .

# =========================
# Install Python Dependencies
# =========================
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# Copy Backend App Files
# =========================
COPY Backend/app/ .

# =========================
# Expose Port
# =========================
EXPOSE 8000

# =========================
# Start App
# =========================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]