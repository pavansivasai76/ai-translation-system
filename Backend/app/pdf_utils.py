import io
import re
import pdfplumber
import fitz  # pymupdf
import numpy as np
import cv2
from typing import List, Dict

from app.ocr_utils import extract_text_from_image


# -------------------------------------------------
# CONFIGURATION (NEW)
# -------------------------------------------------
MAX_PAGES_ALLOWED = 20       # Hard cap to prevent freeze
OCR_DPI = 200                # Reduced from 300 → saves memory
MIN_TEXT_LENGTH = 30


# -------------------------------------------------
# TEXT QUALITY CHECKS
# -------------------------------------------------
def is_corrupted_text(text: str) -> bool:
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return True

    total_len = max(len(text), 1)

    digit_ratio = sum(c.isdigit() for c in text) / total_len
    zero_ratio = text.count("0") / total_len
    replacement_chars = text.count("�")

    noise = len(re.findall(r"[{}\[\]()<>/\\|=_~•■◆▪%$₹€]", text))
    letters = len(re.findall(r"[A-Za-z\u0900-\u097F]", text))

    if digit_ratio > 0.30:
        return True
    if zero_ratio > 0.20:
        return True
    if replacement_chars > 2:
        return True
    if letters == 0 or noise > letters:
        return True

    return False


def is_layout_garbage(text: str) -> bool:
    if not text:
        return True

    total = len(text)
    digits = sum(c.isdigit() for c in text)
    zeros = text.count("0")

    if total == 0:
        return True
    if digits / total > 0.25:
        return True
    if zeros / total > 0.15:
        return True

    return False


def has_reasonable_line_structure(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 5:
        return False

    avg_len = sum(len(l) for l in lines) / len(lines)
    return avg_len < 120


# -------------------------------------------------
# SMART OCR DECISION (NEW)
# -------------------------------------------------
def should_use_ocr(text: str) -> bool:
    """
    Decide if OCR is really needed.
    """
    if is_layout_garbage(text):
        return True

    if is_corrupted_text(text):
        return True

    if not has_reasonable_line_structure(text):
        return True

    return False


# -------------------------------------------------
# MAIN PDF EXTRACTION (OPTIMIZED)
# -------------------------------------------------
def extract_pdf_pages(pdf_bytes: bytes) -> List[Dict]:

    pages_output: List[Dict] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:

        total_pages = len(pdf.pages)

        # -------- PAGE LIMIT PROTECTION --------
        if total_pages > MAX_PAGES_ALLOWED:
            raise ValueError(
                f"PDF too long. Maximum {MAX_PAGES_ALLOWED} pages allowed."
            )

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        try:
            for i, page in enumerate(pdf.pages):

                page_number = i + 1
                extracted_text = page.extract_text() or ""

                # -------- DECIDE OCR OR NOT --------
                if not should_use_ocr(extracted_text):
                    pages_output.append({
                        "page_number": page_number,
                        "text": extracted_text.strip()
                    })
                    continue

                # -------- OCR FALLBACK --------
                pdf_page = doc[i]

                # Reduced DPI (major performance gain)
                pix = pdf_page.get_pixmap(dpi=OCR_DPI)

                img = np.frombuffer(pix.samples, dtype=np.uint8)
                img = img.reshape(pix.height, pix.width, pix.n)

                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif pix.n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                success, buffer = cv2.imencode(".png", img)
                if not success:
                    continue

                ocr_text = extract_text_from_image(
                    buffer.tobytes(),
                    force_indic=True
                )

                if ocr_text and len(ocr_text.strip()) > MIN_TEXT_LENGTH:
                    pages_output.append({
                        "page_number": page_number,
                        "text": ocr_text.strip()
                    })

        finally:
            doc.close()

    return pages_output
