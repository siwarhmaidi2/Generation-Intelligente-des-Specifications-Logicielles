import argparse
import re
from pathlib import Path
from typing import List

from agent_analyst import AgentAnalyst
from agent_analyst.document_loader import extract_text_from_document
from agent_analyst.chunking import ChunkingConfig, split_text_words, merge_analyses


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# =============================
# 🔧 OCR CLEANER (IMPORTANT)
# =============================
def clean_ocr_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[|_•·]", " ", text)
    text = re.sub(r"([a-zA-Z])\s+([a-zA-Z])", r"\1\2", text)  # lettres séparées
    return text.strip()


# =============================
# 📁 Image iterator
# =============================
def iter_images(input_dir: Path, recursive: bool) -> List[Path]:
    files = (
        list(input_dir.rglob("*")) if recursive else list(input_dir.iterdir())
    )
    return sorted([p for p in files if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])


# =============================
# 🚀 MAIN
# =============================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR images -> analyse besoins -> JSON structuré"
    )
    parser.add_argument("--input-dir", type=str, default="images")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lang", type=str, default="fra")

    # Chunking optimisé OCR
    parser.add_argument("--chunk-words", type=int, default=300)
    parser.add_argument("--overlap-words", type=int, default=30)
    parser.add_argument("--min-words-to-chunk", type=int, default=400)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    images = iter_images(input_dir, args.recursive)
    if not images:
        print(f"❌ Aucune image trouvée dans {input_dir}")
        return

    agent = AgentAnalyst()

    cfg = ChunkingConfig(
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        min_words_to_chunk=args.min_words_to_chunk,
    )

    for img in images:
        out = output_dir / f"{img.stem}.json"

        if out.exists() and not args.overwrite:
            print(f"⏭️ Skip : {img.name}")
            continue

        print(f"🔍 OCR : {img.name}")
        text = extract_text_from_document(img, ocr_lang=args.lang)

        if not text.strip():
            print(f"⚠️ OCR vide : {img.name}")
            continue

        text = clean_ocr_text(text)

        chunks = split_text_words(text, cfg)

        print(f"🧩 Chunks : {len(chunks)}")

        if len(chunks) == 1:
            analysis = agent.analyze_text(text)
        else:
            analyses = [agent.analyze_text(c) for c in chunks]
            analysis = merge_analyses(analyses)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            analysis.model_dump_json(indent=2, ensure_ascii=False, exclude={"raw_model_output"}),
            encoding="utf-8",
        )

        print(f"✅ OK : {out}")

    print("\n🎉 Traitement terminé")


if __name__ == "__main__":
    main()
