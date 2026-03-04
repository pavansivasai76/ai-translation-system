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
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-nep \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && which tesseract \
    && tesseract --version \
    && rm -rf /var/lib/apt/lists/*

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# =========================
# Set Work Directory
# =========================
WORKDIR /app

# =========================
# Copy Requirements
# =========================
COPY Backend/requirements.txt .

# =========================
# Install Python Dependencies
# =========================
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# Copy Backend App
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