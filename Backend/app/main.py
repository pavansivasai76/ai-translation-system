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
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastapi.responses import StreamingResponse
from io import BytesIO
from docx import Document


# =========================================================
# LOAD ENV VARIABLES
# =========================================================
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY not set in environment variables")


# =========================================================
# APP INIT
# =========================================================
app = FastAPI(title="AI Translation System")

Base.metadata.create_all(bind=engine)


# =========================================================
# CORS CONFIG  (FIXED FOR VERCEL)
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


# =========================================================
# ERROR HANDLERS
# =========================================================
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
# RATE LIMITING
# =========================================================
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW = 60

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
# RATE LIMIT CHECK
# =========================================================
def check_rate_limit(user_email: str):
    now = datetime.utcnow().timestamp()

    if user_email not in user_request_log:
        user_request_log[user_email] = []

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
# STARTUP
# =========================================================
@app.on_event("startup")
def startup_event():
    warmup_model()


# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():
    return {"message": "AI Translation System API is running"}