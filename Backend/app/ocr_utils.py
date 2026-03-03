import cv2
import pytesseract
import numpy as np
from PIL import Image
from io import BytesIO
import re
import os

# ---------------------------------------------------------
# TESSERACT PATH (Windows)
# ---------------------------------------------------------
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


# ---------------------------------------------------------
# BASIC OCR TEXT CLEANING (NEWLINE-SAFE)
# ---------------------------------------------------------
def clean_ocr_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"_+", " ", text)
    text = re.sub(r"(?:\b[A-Za-z]\b\s*/\s*)+", " ", text)
    text = re.sub(r"[|~•■◆▪]", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()


# ---------------------------------------------------------
# REMOVE GARBAGE LINES (SCRIPT + DENSITY AWARE)
# ---------------------------------------------------------
def remove_garbage_lines(text: str) -> str:
    if not text:
        return ""

    clean_lines = []

    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) < 3:
            continue

        devanagari = len(re.findall(r"[\u0900-\u097F]", line))
        latin = len(re.findall(r"[A-Za-z]", line))
        digits = len(re.findall(r"[0-9]", line))
        noise = len(re.findall(r"[{}\[\]()<>/\\|=_~•■◆▪%$₹€]", line))

        if devanagari >= 3 and devanagari > digits + noise:
            clean_lines.append(line)
            continue

        if latin >= 5 and latin > digits + noise:
            clean_lines.append(line)
            continue

    return "\n".join(clean_lines)


# ---------------------------------------------------------
# OCR EXTRACTION (COLOR-ROBUST VERSION)
# ---------------------------------------------------------
def extract_text_from_image(
    image_bytes: bytes,
    force_indic: bool = False
) -> str:

    # -----------------------------
    # Bytes → PIL → OpenCV
    # -----------------------------
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # -----------------------------
    # Upscale (critical for Devanagari)
    # -----------------------------
    img = cv2.resize(
        img,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC
    )

    # =====================================================
    # COLOR-ROBUST PREPROCESSING
    # =====================================================

    # -----------------------------
    # LAB color space (better than grayscale)
    # -----------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # -----------------------------
    # Convert to grayscale
    # -----------------------------
    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # Light noise reduction
    # -----------------------------
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # -----------------------------
    # Threshold Variant 1 (Otsu)
    # -----------------------------
    _, thresh1 = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # -----------------------------
    # Threshold Variant 2 (Adaptive Gaussian)
    # -----------------------------
    thresh2 = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    # -----------------------------
    # Threshold Variant 3 (Inverted)
    # -----------------------------
    thresh3 = cv2.bitwise_not(thresh2)

    # -----------------------------
    # Morphological strengthening
    # -----------------------------
    kernel = np.ones((2, 2), np.uint8)

    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    thresh2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    thresh3 = cv2.morphologyEx(thresh3, cv2.MORPH_CLOSE, kernel)

    # =====================================================
    # OCR CONFIG
    # =====================================================
    if force_indic:
        custom_config = (
            "--oem 3 "
            "--psm 6 "
            "-l hin+nep "
            "-c tessedit_char_blacklist="
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
    else:
        custom_config = "--oem 3 --psm 6 -l hin+nep+eng"

    # =====================================================
    # MULTI-PASS OCR (Pick Best Result)
    # =====================================================
    candidates = []

    for variant in [thresh1, thresh2, thresh3]:
        txt = pytesseract.image_to_string(
            variant,
            config=custom_config
        )
        candidates.append(txt)

    # Also try enhanced grayscale directly
    txt_gray = pytesseract.image_to_string(
        gray,
        config=custom_config
    )
    candidates.append(txt_gray)

    # Pick best candidate (longest clean text)
    text = max(
        candidates,
        key=lambda x: len(x.strip())
    )

    # =====================================================
    # CLEAN + FILTER OCR OUTPUT
    # =====================================================
    text = clean_ocr_text(text)
    text = remove_garbage_lines(text)

    return text
