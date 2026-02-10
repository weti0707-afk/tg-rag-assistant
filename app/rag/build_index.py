from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.rag.ingest import load_and_chunk_dir
from app.rag.index import RagIndex


def main() -> None:
    s = get_settings()
    materials_dir = Path("data/materials")
    index_dir = Path("data/index")

    chunks = load_and_chunk_dir(materials_dir, chunk_size=s.rag_chunk_size, overlap=s.rag_chunk_overlap)
    rag = RagIndex(model_name=s.embedding_model)
    rag.build(chunks)
    rag.save(index_dir)

    print(f"INDEX_OK chunks={len(chunks)} index_dir={index_dir}")


if __name__ == "__main__":
    main()
