import argparse
from pathlib import Path

from agent_analyst import AgentAnalyst
from agent_analyst.document_loader import extract_text_from_document
from agent_analyst.chunking import ChunkingConfig, split_text_words, merge_analyses


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_images(input_dir: Path, recursive: bool) -> list[Path]:
    files = [p for p in (input_dir.rglob("*") if recursive else input_dir.iterdir()) if p.is_file()]
    return sorted([p for p in files if p.suffix.lower() in SUPPORTED_EXTS])


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch: images/ -> results/ (OCR puis analyse, JSON).")
    parser.add_argument("--input-dir", type=str, default="images")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lang", type=str, default="fra")
    parser.add_argument("--chunk-words", type=int, default=450)
    parser.add_argument("--overlap-words", type=int, default=60)
    parser.add_argument("--min-words-to-chunk", type=int, default=600)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    images = iter_images(input_dir, recursive=args.recursive)
    if not images:
        print(f"Aucune image dans {input_dir}")
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
            print(f"⏭️  Skip (existe déjà) : {out}")
            continue

        # OCR (extract_text_from_document gère images)
        text = extract_text_from_document(img, ocr_lang=args.lang)
        chunks = split_text_words(text, cfg)
        if len(chunks) == 1:
            analysis = agent.analyze_text(text)
        else:
            analyses = [agent.analyze_text(c) for c in chunks]
            analysis = merge_analyses(analyses)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(analysis.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✅ OK : {img.name} -> {out}")


if __name__ == "__main__":
    main()


