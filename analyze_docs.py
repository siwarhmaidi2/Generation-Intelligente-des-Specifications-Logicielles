import argparse
from pathlib import Path

from agent_analyst import AgentAnalyst
from agent_analyst.document_loader import detect_kind, extract_text_from_document
from agent_analyst.chunking import ChunkingConfig, split_text_words, merge_analyses


SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_documents(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file()]

    docs: list[Path] = []
    for p in files:
        if p.suffix.lower() in SUPPORTED_EXTS:
            docs.append(p)

    return sorted(docs)


def analyze_one(
    agent: AgentAnalyst,
    input_path: Path,
    output_path: Path,
    ocr_lang: str,
    overwrite: bool,
    chunk_words: int,
    overlap_words: int,
    min_words_to_chunk: int,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"⏭️  Skip (existe déjà) : {output_path}")
        return

    # Valider format (erreur explicite si extension non supportée)
    _ = detect_kind(input_path)

    text = extract_text_from_document(input_path, ocr_lang=ocr_lang)
    cfg = ChunkingConfig(
        chunk_words=chunk_words,
        overlap_words=overlap_words,
        min_words_to_chunk=min_words_to_chunk,
    )

    chunks = split_text_words(text, cfg)
    if len(chunks) == 1:
        analysis = agent.analyze_text(text)
    else:
        print(f"🧩 Chunking: {len(chunks)} morceaux (~{chunk_words} mots, overlap {overlap_words})")
        analyses = []
        for i, chunk in enumerate(chunks, start=1):
            print(f"  - Chunk {i}/{len(chunks)} ...")
            analyses.append(agent.analyze_text(chunk))
        analysis = merge_analyses(analyses)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        analysis.model_dump_json(indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"✅ OK : {input_path.name} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyser tous les PDF/images d’un dossier docs/ (OCR si nécessaire) et écrire les JSON dans results/."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="docs",
        help="Dossier d’entrée contenant PDF/images (défaut: docs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Dossier de sortie pour les JSON (défaut: results)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Parcourir récursivement le dossier d’entrée",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Écraser les JSON déjà existants",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="fra",
        help="Langue Tesseract (ex: fra, eng, fra+eng). Défaut: fra",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=450,
        help="Taille d’un chunk en mots (défaut: 450).",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=60,
        help="Overlap entre chunks en mots (défaut: 60).",
    )
    parser.add_argument(
        "--min-words-to-chunk",
        type=int,
        default=600,
        help="Seuil (en mots) à partir duquel on active le chunking (défaut: 600).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Dossier introuvable: {input_dir}")

    documents = iter_documents(input_dir, recursive=args.recursive)
    if not documents:
        print(f"Aucun document trouvé dans {input_dir} (extensions: {', '.join(sorted(SUPPORTED_EXTS))})")
        return

    agent = AgentAnalyst()

    print(f"📂 Entrée : {input_dir}")
    print(f"📁 Sortie : {output_dir}")
    print(f"📄 Fichiers: {len(documents)}")
    print(f"🔤 OCR lang: {args.lang}")
    print(
        f"🧩 Chunking: chunk_words={args.chunk_words}, overlap_words={args.overlap_words}, min_words_to_chunk={args.min_words_to_chunk}"
    )
    print("")

    for doc in documents:
        out = output_dir / f"{doc.stem}.json"
        try:
            analyze_one(
                agent,
                doc,
                out,
                ocr_lang=args.lang,
                overwrite=args.overwrite,
                chunk_words=args.chunk_words,
                overlap_words=args.overlap_words,
                min_words_to_chunk=args.min_words_to_chunk,
            )
        except Exception as e:
            print(f"❌ ERREUR : {doc} -> {e}")


if __name__ == "__main__":
    main()


