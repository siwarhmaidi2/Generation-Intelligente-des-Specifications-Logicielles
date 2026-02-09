from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from .schemas import RequirementsAnalysis, Requirement, Actor, BusinessRule


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_words: int = 450
    overlap_words: int = 60
    min_words_to_chunk: int = 600


def _normalize_text(text: str) -> str:
    # Nettoyage léger pour stabiliser le découpage
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Réduire les espaces
    text = re.sub(r"[ \t]+", " ", text)
    # Réduire les lignes vides multiples
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_words(text: str, cfg: ChunkingConfig) -> List[str]:
    """
    Découpe un texte en chunks (en mots) avec un overlap.
    Essaie de couper à des séparateurs naturels (double saut de ligne / fin de phrase) si possible.
    """
    text = _normalize_text(text)
    words = text.split()
    if len(words) <= cfg.min_words_to_chunk:
        return [text]

    chunks: List[str] = []
    step = max(1, cfg.chunk_words - cfg.overlap_words)

    i = 0
    while i < len(words):
        window = words[i : i + cfg.chunk_words]
        chunk = " ".join(window)

        # Tentative de coupe naturelle dans les ~25% finaux
        cut_idx = None
        tail_start = int(len(chunk) * 0.75)
        for sep in ["\n\n", ". ", "; ", ": "]:
            pos = chunk.rfind(sep, tail_start)
            if pos != -1:
                cut_idx = pos + len(sep)
                break
        if cut_idx and cut_idx > 0:
            chunk = chunk[:cut_idx].strip()

        chunks.append(chunk)
        i += step

    # Dé-dupliquer les chunks identiques (rare mais possible)
    deduped: List[str] = []
    seen = set()
    for c in chunks:
        key = c[:200]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def _renumber(prefix: str, idx: int) -> str:
    return f"{prefix}{idx}"


def merge_analyses(analyses: Iterable[RequirementsAnalysis]) -> RequirementsAnalysis:
    """
    Fusionne plusieurs analyses (chunks) en une seule.
    - Dé-duplique acteurs par nom
    - Dé-duplique règles par description
    - Renumérote FR/NFR/BR proprement
    """
    merged = RequirementsAnalysis(
        functional_requirements=[],
        non_functional_requirements=[],
        actors=[],
        business_rules=[],
        summary="",
        metadata={"chunked": "true"},
    )

    # Collect
    fr_desc_seen = set()
    nfr_desc_seen = set()
    br_desc_seen = set()
    actor_name_seen = set()

    fr: List[Requirement] = []
    nfr: List[Requirement] = []
    br: List[BusinessRule] = []
    actors: List[Actor] = []
    summaries: List[str] = []

    for a in analyses:
        if a.summary:
            summaries.append(a.summary.strip())

        for r in a.functional_requirements:
            key = r.description.strip().lower()
            if key and key not in fr_desc_seen:
                fr_desc_seen.add(key)
                fr.append(r)

        for r in a.non_functional_requirements:
            key = r.description.strip().lower()
            if key and key not in nfr_desc_seen:
                nfr_desc_seen.add(key)
                nfr.append(r)

        for x in a.business_rules:
            key = x.description.strip().lower()
            if key and key not in br_desc_seen:
                br_desc_seen.add(key)
                br.append(x)

        for ac in a.actors:
            key = ac.name.strip().lower()
            if key and key not in actor_name_seen:
                actor_name_seen.add(key)
                actors.append(ac)

    # Renumérotation
    for i, r in enumerate(fr, start=1):
        merged.functional_requirements.append(
            Requirement(
                id=_renumber("FR", i),
                description=r.description,
                type="fonctionnel",
                priority=r.priority,
            )
        )

    for i, r in enumerate(nfr, start=1):
        merged.non_functional_requirements.append(
            Requirement(
                id=_renumber("NFR", i),
                description=r.description,
                type="non_fonctionnel",
                priority=r.priority,
            )
        )

    for i, x in enumerate(br, start=1):
        merged.business_rules.append(
            BusinessRule(
                id=_renumber("BR", i),
                description=x.description,
                explicit=x.explicit,
            )
        )

    merged.actors = actors

    # Résumé: on garde les 3 premiers résumés concaténés (évite un pavé)
    merged.summary = " ".join(summaries[:3]).strip()
    return merged



