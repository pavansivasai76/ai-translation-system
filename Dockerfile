# ================================
# Base Image (Python 3.11)
# ================================
FROM python:3.11-slim

# ================================
# Prevent Python buffering
# ================================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ================================
# System Dependencies
# ================================
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-hin \
    tesseract-ocr-nep \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Set Work Directory
# ================================
WORKDIR /app

# ================================
# Copy Requirements First
# ================================
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ================================
# Copy App Code
# ================================
COPY . .

# ================================
# Expose Port
# ================================
EXPOSE 8000

# ================================
# Start FastAPI
# ================================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]