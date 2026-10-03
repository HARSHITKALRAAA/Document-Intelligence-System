"""PDF ingestion, page-aware chunking, and cosine vector retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import math
import re
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Passage:
    document: str
    page: int
    text: str
    chunk_id: str


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


def extract_pdf_pages(filename: str, content: bytes) -> list[tuple[str, int, str]]:
    """Return searchable text with the original one-based PDF page number."""
    import fitz

    if not content.startswith(b"%PDF-"):
        raise ValueError(f"{filename}: this file is not a PDF.")
    try:
        with fitz.open(stream=BytesIO(content), filetype="pdf") as pdf:
            if pdf.is_encrypted:
                raise ValueError(f"{filename}: password-protected PDFs are unsupported.")
            pages = [(filename, i + 1, page.get_text(sort=True)) for i, page in enumerate(pdf)]
    except (fitz.FileDataError, fitz.EmptyFileError) as exc:
        raise ValueError(f"{filename}: the PDF could not be read.") from exc
    if not any(text.strip() for _, _, text in pages):
        raise ValueError(f"{filename}: no selectable text was found. Scanned PDFs need OCR.")
    return pages


def chunk_pages(
    pages: Sequence[tuple[str, int, str]], *, size: int = 900, overlap: int = 150
) -> list[Passage]:
    """Split within each page, retaining the source page for citations."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must exceed a nonnegative overlap.")
    result: list[Passage] = []
    for filename, page, raw in pages:
        clean = re.sub(r"\s+", " ", raw).strip()
        start = 0
        while start < len(clean):
            end = min(start + size, len(clean))
            if end < len(clean):
                boundary = clean.rfind(" ", start + size // 2, end)
                if boundary > start:
                    end = boundary
            piece = clean[start:end].strip()
            if piece:
                digest = hashlib.sha256(f"{filename}|{page}|{start}|{piece}".encode()).hexdigest()[:16]
                result.append(Passage(filename, page, piece, digest))
            if end >= len(clean):
                break
            start = max(start + 1, end - overlap)
    return result


def _unit(vector: Sequence[float]) -> list[float]:
    length = math.sqrt(sum(float(n) ** 2 for n in vector))
    return [float(n) / length for n in vector] if length else [0.0 for _ in vector]


class VectorIndex:
    def __init__(self, passages: Sequence[Passage], encoder: Encoder):
        if not passages:
            raise ValueError("No searchable text was extracted from the PDFs.")
        self.passages = list(passages)
        self.encoder = encoder
        self.vectors = [_unit(v) for v in encoder.encode([p.text for p in passages])]
        if len(self.vectors) != len(self.passages):
            raise ValueError("The embedding model returned the wrong number of vectors.")

    def search(self, question: str, *, top_k: int = 5, documents: set[str] | None = None) -> list[Hit]:
        if not question.strip():
            return []
        query_vectors = self.encoder.encode([question])
        if len(query_vectors) != 1:
            raise ValueError("The embedding model did not return a query vector.")
        query = _unit(query_vectors[0])
        hits = []
        for passage, vector in zip(self.passages, self.vectors):
            if documents is not None and passage.document not in documents:
                continue
            if len(vector) != len(query):
                raise ValueError("Embedding dimensions differ between document and query.")
            hits.append(Hit(passage, sum(a * b for a, b in zip(vector, query))))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:max(0, top_k)]
