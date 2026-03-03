# AI Translation System

AI-powered translation system supporting:
- Nepali to English
- Hindi to English
- OCR from images
- PDF document translation

## Features
- JWT Authentication
- Rate Limiting
- OCR with Tesseract
- Neural Machine Translation (IndicTrans2)
- PDF Extraction + Structure Cleaning
- TXT & DOCX Download

## Setup

1. Clone repo
2. Create virtual environment
3. Install dependencies:
   pip install -r requirements.txt
4. Create .env file:

SECRET_KEY=your_secret_key
DATABASE_URL=postgresql://user:pass@localhost/db
FRONTEND_ORIGIN=http://127.0.0.1:5500

5. Run:
   uvicorn main:app --reload