from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader


SUPPORTED_EXT = {".txt", ".pdf", ".docx"}


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n".join(parts)


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts)


def read_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return read_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    # простое чанкирование по символам (для MVP нормально)
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def load_and_chunk_dir(materials_dir: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in sorted(materials_dir.rglob("*")):
        if p.is_dir():
            continue
        if p.suffix.lower() not in SUPPORTED_EXT:
            continue
        raw = read_document(p)
        parts = chunk_text(raw, chunk_size=chunk_size, overlap=overlap)
        doc_id = p.relative_to(materials_dir).as_posix()
        for i, t in enumerate(parts):
            chunks.append(Chunk(doc_id=doc_id, chunk_id=i, text=t))
    return chunks
