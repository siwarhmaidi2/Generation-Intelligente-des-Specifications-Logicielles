import os
from pathlib import Path

import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
def configure_tesseract_cmd() -> None:
    cmd = os.getenv("TESSERACT_CMD", "").strip()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def ocr_image(image_path: str | Path, lang: str = "fra") -> str:
    configure_tesseract_cmd()
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image introuvable : {path}")
    with Image.open(path) as img:
        text = pytesseract.image_to_string(img, lang=lang)
    return (text or "").strip()


def ocr_pil_image(img: Image.Image, lang: str = "fra") -> str:
    configure_tesseract_cmd()
    text = pytesseract.image_to_string(img, lang=lang)
    return (text or "").strip()


def looks_like_text_is_insufficient(text: str, min_chars: int = 400) -> bool:
    return len((text or "").strip()) < min_chars


