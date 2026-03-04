this is is my entire main.py
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy.orm import Session
from app.database import engine
from app.models import Base

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, LangDetectException
import re
import os
from dotenv import load_dotenv

from app.database import SessionLocal
from app.models import User
from passlib.context import CryptContext

from jose import JWTError, jwt
from datetime import datetime, timedelta

from app.ocr_utils import extract_text_from_image
from app.pdf_utils import extract_pdf_pages

from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.exceptions import RequestValidationError
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastapi.responses import StreamingResponse
from io import BytesIO
from docx import Document


# =========================================================
# LOAD ENV VARIABLES (SECURITY FIX)
# =========================================================
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://127.0.0.1:5500")

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY not set in environment variables")

# =========================================================
# APP INIT
# =========================================================
app = FastAPI(title="AI Translation System")

Base.metadata.create_all(bind=engine)
# =========================================================
# CORS CONFIG
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-translation-system.vercel.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error_code": "VALIDATION_ERROR",
            "message": "Invalid request payload"
        },
    )



@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "Something went wrong."
        },
    )

# =========================================================
# PASSWORD HASHING
# =========================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# =========================================================
# SECURITY
# =========================================================
security = HTTPBearer()


# =========================================================
# DATABASE DEPENDENCY
# =========================================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================================================
# FILE SIZE LIMITS
# =========================================================
MAX_IMAGE_SIZE_MB = 5
MAX_PDF_SIZE_MB = 10

MAX_IMAGE_SIZE = MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_PDF_SIZE = MAX_PDF_SIZE_MB * 1024 * 1024

# =========================================================
# DOWNLOAD SIZE LIMIT
# =========================================================
MAX_DOWNLOAD_SIZE_MB = 2
MAX_DOWNLOAD_SIZE = MAX_DOWNLOAD_SIZE_MB * 1024 * 1024


# =========================================================
# RATE LIMITING CONFIG
# =========================================================
RATE_LIMIT_REQUESTS = 20        # Max requests
RATE_LIMIT_WINDOW = 60          # Per 60 seconds

user_request_log = {}


# =========================================================
# TOKEN CREATION
# =========================================================
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()

    expire = datetime.utcnow() + (
        expires_delta if expires_delta
        else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


# =========================================================
# TOKEN VERIFICATION
# =========================================================
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")

        if email is None:
            raise HTTPException(status_code=401, detail="INVALID_TOKEN")

    except JWTError:
        raise HTTPException(status_code=401, detail="INVALID_TOKEN")

    user = db.query(User).filter(User.email == email).first()

    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user

# =========================================================
# RATE LIMIT CHECKER
# =========================================================
def check_rate_limit(user_email: str):
    now = datetime.utcnow().timestamp()

    if user_email not in user_request_log:
        user_request_log[user_email] = []

    # Keep only requests inside window
    user_request_log[user_email] = [
        t for t in user_request_log[user_email]
        if now - t < RATE_LIMIT_WINDOW
    ]

    if len(user_request_log[user_email]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making more requests."
        )

    user_request_log[user_email].append(now)

# =========================================================
# LOAD MODEL
# =========================================================
MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
).to(device)

model.eval()

def warmup_model():
    dummy_text = "hin_Deva eng_Latn नमस्ते दुनिया"
    inputs = tokenizer(dummy_text, return_tensors="pt").to(device)

    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5)


# =========================================================
# LANGUAGE MAP
# =========================================================
LANG_MAP = {
    "hi": "hin_Deva",
    "ne": "npi_Deva",
    "en": "eng_Latn"
}


# =========================================================
# SCHEMAS
# =========================================================
class TranslateRequest(BaseModel):
    text: str
    source_lang: Optional[str] = None


class TranslateResponse(BaseModel):
    translated_text: str
    detected_language: str


class SignupRequest(BaseModel):
    full_name: str
    email: str
    password: str


class SignupResponse(BaseModel):
    message: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class ErrorResponse(BaseModel):
    success: bool
    error_code: str
    message: str
# =========================================================
# SIGNUP API
# =========================================================
@app.post("/signup", response_model=SignupResponse)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):

    existing_user = db.query(User).filter(User.email == payload.email).first()

    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    safe_password = payload.password[:72]
    hashed_password = pwd_context.hash(safe_password)

    new_user = User(
        full_name=payload.full_name,
        email=payload.email,
        hashed_password=hashed_password
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User registered successfully"}


# =========================================================
# LOGIN API
# =========================================================
@app.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):

    user = db.query(User).filter(User.email == payload.email).first()

    if not user:
        raise HTTPException(status_code=400, detail="INVALID_CREDENTIALS")

    if not pwd_context.verify(payload.password[:72], user.hashed_password):
        raise HTTPException(status_code=400, detail="INVALID_CREDENTIALS")

    access_token = create_access_token(
        data={"sub": user.email}
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.on_event("startup")
def startup_event():
    warmup_model()


# =========================================================
# LANGUAGE DETECTION
# =========================================================
def is_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))


def detect_hi_ne(text: str) -> str:
    nepali_markers = ["छ", "छु", "छैन", "थियो", "गर्छ", "हुन्छ", "लाई", "बाट"]
    hindi_markers = ["है", "हैं", "था", "थे", "करता", "किया", "को", "से"]

    nepali_score = sum(text.count(w) for w in nepali_markers)
    hindi_score = sum(text.count(w) for w in hindi_markers)

    return "ne" if nepali_score > hindi_score else "hi"


def detect_language(text: str) -> str:
    if is_devanagari(text):
        return detect_hi_ne(text)

    try:
        if detect(text) == "en":
            return "en"
    except LangDetectException:
        pass

    return "unknown"

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": device
    }

# =========================================================
# TRANSLATION CORE
# =========================================================
def translate_core(text: str) -> tuple[str, str]:

    text = text.strip()

    if not text or len(text) < 5:
        return "", "unknown"

    # Hard cap input length
    text = text[:4000]

    src_lang = detect_language(text)

    if src_lang not in ("hi", "ne", "en"):
        return "", "unknown"

    if src_lang == "en":
        return text, "en"

    # Split into sentences
    sentences = [
        s.strip()
        for s in re.split(r'(?<=[।.!?])\s+', text)
        if s.strip()
    ]

    if not sentences:
        return "", "unknown"

    # Add language tags
    tagged_texts = [
        f"{LANG_MAP[src_lang]} {LANG_MAP['en']} {sentence}"
        for sentence in sentences
    ]

    # Batch tokenize
    inputs = tokenizer(
        tagged_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,      # dynamic generation length
            num_beams=2,             # reduced beams (faster)
            early_stopping=True,
            use_cache=True
        )

    # Decode batch output
    translated_sentences = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    translated_text = " ".join(translated_sentences)

    return translated_text.strip(), src_lang


# =========================================================
# BLOCK SPLITTER
# =========================================================
def split_blocks(text: str) -> List[str]:
    markers = [
        "Information",
        "Notice",
        "Announcement",
        "Organized",
        "Date",
        "Head Boy",
        "Principal",
        "सूचना",
        "विद्यालय",
        "दिनांक"
    ]

    blocks = []
    current = ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if any(m.lower() in line.lower() for m in markers) and current:
            blocks.append(current.strip())
            current = line
        else:
            current += " " + line

    if current:
        blocks.append(current.strip())

    return blocks


# =========================================================
# BEAUTIFY
# =========================================================
def beautify_translation(text: str) -> str:

    two_group_patterns = [
        r"(Information|Notice|Announcement)\s+(\d{1,2}\s+\w+\s+20XX)",
        r"(Information|Notice)\s*:\s*(\w+\s+\d{1,2},\s+20XX)"
    ]

    one_group_patterns = [
        r"(Awareness campaign on cleanliness)",
        r"(Khadi Textiles Discount Announcement)",
        r"(The change in school time)",
        r"(of the Poet's Conference)"
    ]

    for p in two_group_patterns:
        text = re.sub(p, r"\1\n\2", text, flags=re.IGNORECASE)

    for p in one_group_patterns:
        text = re.sub(p, r"\n\1\n", text, flags=re.IGNORECASE)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================================================
# PROTECTED TRANSLATE API
# =========================================================
@app.post("/translate", response_model=TranslateResponse)
def translate_text(
    payload: TranslateRequest,
    current_user: User = Depends(get_current_user)
):
    check_rate_limit(current_user.email)
    translated_text, detected_lang = translate_core(payload.text)
    return {
        "translated_text": translated_text,
        "detected_language": detected_lang
    }


# (Everything above remains EXACTLY the same — unchanged)

# =========================================================
# PROTECTED OCR API
# =========================================================
@app.post("/ocr")
async def ocr_translate(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    check_rate_limit(current_user.email)
    # Validate file type
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image format")

    image_bytes = await file.read()

    # Validate file size
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large. Max size {MAX_IMAGE_SIZE_MB}MB"
        )

    extracted_text = extract_text_from_image(image_bytes)

    if not extracted_text:
        return {
            "extracted_text": "",
            "translated_text": "No readable text detected in image"
        }

    translated_text, detected_lang = translate_core(extracted_text)

    return {
        "extracted_text": extracted_text,
        "translated_text": translated_text,
        "detected_language": detected_lang
    }


# =========================================================
# PROTECTED PDF API
# =========================================================
@app.post("/pdf")
async def pdf_translate(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    check_rate_limit(current_user.email)
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid PDF format")

    pdf_bytes = await file.read()

    # Validate file size
    if len(pdf_bytes) > MAX_PDF_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"PDF too large. Max size {MAX_PDF_SIZE_MB}MB"
        )

    pages = extract_pdf_pages(pdf_bytes)

    if not pages:
        return {
            "pages": [],
            "message": "No readable text detected in PDF"
        }

    translated_pages = []

    for page in pages:
        raw_text = page.get("text", "").strip()
        if not raw_text or len(raw_text) < 10:
            continue

        blocks = split_blocks(raw_text)
        translated_blocks = []

        for block in blocks:
            translated, detected_lang = translate_core(block)
            if translated:
                translated_blocks.append(
                    beautify_translation(translated)
                )

        translated_pages.append({
            "page_number": page["page_number"],
            "extracted_text": raw_text,
            "translated_text": "\n\n".join(translated_blocks),
            "detected_language": detected_lang
        })

    return {
        "total_pages_processed": len(translated_pages),
        "pages": translated_pages
    }
@app.post("/download/txt")
def download_txt(
    content: dict,
    current_user: User = Depends(get_current_user)
):
    check_rate_limit(current_user.email)

    text = content.get("text", "")

    if not text:
        raise HTTPException(status_code=400, detail="No content provided")

    if len(text.encode("utf-8")) > MAX_DOWNLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Download size exceeds {MAX_DOWNLOAD_SIZE_MB}MB limit"
        )

    buffer = BytesIO()
    buffer.write(text.encode("utf-8"))
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/plain",
        headers={
            "Content-Disposition": "attachment; filename=translation.txt"
        }
    )

@app.post("/download/docx")
def download_docx(
    content: dict,
    current_user: User = Depends(get_current_user)
):
    check_rate_limit(current_user.email)

    text = content.get("text", "")

    if not text:
        raise HTTPException(status_code=400, detail="No content provided")

    if len(text.encode("utf-8")) > MAX_DOWNLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Download size exceeds {MAX_DOWNLOAD_SIZE_MB}MB limit"
        )

    document = Document()
    document.add_paragraph(text)

    buffer = BytesIO()
    document.save(buffer)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={
            "Content-Disposition": "attachment; filename=translation.docx"
        }
    )
@app.get("/")
def root():
    return {"message": "AI Translation System API is running"}
