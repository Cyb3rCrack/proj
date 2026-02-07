"""Embedding module.

Uses SentenceTransformers and normalizes embeddings once (cosine-by-contract).
"""

from __future__ import annotations

import numpy as np
import warnings
import os
import sys
import logging
from typing import Iterable, List


class EmbeddingModule:
    def __init__(self):
        self._model = None
        self.model_name = os.getenv("ACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.device = os.getenv("ACE_EMBEDDING_DEVICE", "cpu")
        self.batch_size = int(os.getenv("ACE_EMBEDDING_BATCH_SIZE", "32"))
        self.normalize = os.getenv("ACE_EMBEDDING_NORMALIZE", "true").lower() == "true"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        
        # Suppress warnings during model loading
        import io
        import contextlib
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            
            # Capture and suppress stderr output from HF Hub
            stderr_capture = io.StringIO()
            with contextlib.redirect_stderr(stderr_capture):
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, text: str):
        self._ensure_loaded()
        vec = self._model.encode(text, batch_size=self.batch_size, show_progress_bar=False)
        if not self.normalize:
            return vec
        norm = np.linalg.norm(vec)
        if norm == 0 or np.isnan(norm):
            return vec
        return vec / norm

    def embed_many(self, texts: Iterable[str]) -> List[np.ndarray]:
        self._ensure_loaded()
        vecs = self._model.encode(list(texts), batch_size=self.batch_size, show_progress_bar=False)
        if not self.normalize:
            return [v for v in vecs]
        normalized = []
        for v in vecs:
            norm = np.linalg.norm(v)
            if norm == 0 or np.isnan(norm):
                normalized.append(v)
            else:
                normalized.append(v / norm)
        return normalized


__all__ = ["EmbeddingModule"]
