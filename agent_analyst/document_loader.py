from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF
from PIL import Image

from .ocr import ocr_image, ocr_pil_image, looks_like_text_is_insufficient
from .pdf_loader import extract_text_from_pdf


SupportedKind = Literal["pdf", "image"]


def detect_kind(path: Path) -> SupportedKind:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return "image"
    raise ValueError(f"Format non supporté: {ext} (fichier: {path})")


def extract_text_from_pdf_with_ocr_fallback(
    pdf_path: str | Path,
    ocr_lang: str = "fra",
    min_chars_before_ocr: int = 400,
    dpi: int = 220,
) -> str:
    pdf_path = Path(pdf_path)
    native_text = extract_text_from_pdf(pdf_path)
    if not looks_like_text_is_insufficient(native_text, min_chars=min_chars_before_ocr):
        return native_text

    doc = fitz.open(str(pdf_path))
    texts: list[str] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        texts.append(ocr_pil_image(img, lang=ocr_lang))
    return "\n\n".join(t for t in texts if t).strip()


def extract_text_from_document(
    input_path: str | Path,
    ocr_lang: str = "fra",
    min_chars_before_ocr: int = 400,
) -> str:
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document introuvable : {path}")

    kind = detect_kind(path)
    if kind == "pdf":
        return extract_text_from_pdf_with_ocr_fallback(
            path, ocr_lang=ocr_lang, min_chars_before_ocr=min_chars_before_ocr
        )
    return ocr_image(path, lang=ocr_lang)


