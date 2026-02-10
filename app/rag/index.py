from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

from app.rag.ingest import Chunk


class RagIndex:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedder = TextEmbedding(model_name=model_name)
        self.index: faiss.Index | None = None
        self.meta: list[dict] = []  # {doc_id, chunk_id, text}

    def _embed(self, texts: list[str]) -> np.ndarray:
        # fastembed возвращает итератор numpy-векторов
        vecs = list(self.embedder.embed(texts))
        arr = np.array(vecs, dtype="float32")
        # L2-normalize (для cosine)
        faiss.normalize_L2(arr)
        return arr

    def build(self, chunks: list[Chunk]) -> None:
        self.meta = [asdict(c) for c in chunks]
        texts = [c.text for c in chunks]
        if not texts:
            # пустой индекс
            self.index = faiss.IndexFlatIP(384)  # размерность будет уточнена ниже при первой сборке
            return

        emb = self._embed(texts)
        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine = inner product после нормализации
        self.index.add(emb)

    def search(self, query: str, top_k: int) -> list[dict]:
        if self.index is None or len(self.meta) == 0:
            return []
        q = self._embed([query])
        scores, idxs = self.index.search(q, top_k)
        out: list[dict] = []
        for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i == -1:
                continue
            m = dict(self.meta[i])
            m["score"] = float(score)
            out.append(m)
        return out

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        assert self.index is not None
        faiss.write_index(self.index, str(dir_path / "index.faiss"))
        (dir_path / "meta.json").write_text(json.dumps(self.meta, ensure_ascii=False), encoding="utf-8")
        (dir_path / "model.txt").write_text(self.model_name, encoding="utf-8")

    @classmethod
    def load(cls, dir_path: str | Path) -> "RagIndex":
        dir_path = Path(dir_path)
        model_name = (dir_path / "model.txt").read_text(encoding="utf-8").strip()
        obj = cls(model_name=model_name)
        obj.index = faiss.read_index(str(dir_path / "index.faiss"))
        obj.meta = json.loads((dir_path / "meta.json").read_text(encoding="utf-8"))
        return obj
