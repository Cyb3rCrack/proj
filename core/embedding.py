"""Embedding module.

Uses SentenceTransformers and normalizes embeddings once (cosine-by-contract).
"""

from __future__ import annotations

import numpy as np


class EmbeddingModule:
    def __init__(self):
        self._model = None
        self.model_name = "all-MiniLM-L6-v2"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    def embed(self, text: str):
        self._ensure_loaded()
        vec = self._model.encode(text)
        norm = np.linalg.norm(vec)
        if norm == 0 or np.isnan(norm):
            return vec
        return vec / norm


__all__ = ["EmbeddingModule"]
