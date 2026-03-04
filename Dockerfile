# =========================
# Base Image
# =========================
FROM python:3.11-slim

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
    && rm -rf /var/lib/apt/lists/*

# =========================
# Set Work Directory
# =========================
WORKDIR /app

# =========================
# Copy Requirements
# =========================
COPY Backend/requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# Copy App
# =========================
COPY Backend/app/ .

# =========================
# Expose Port
# =========================
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]