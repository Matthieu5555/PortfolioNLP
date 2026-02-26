"""
Thin wrapper around sentence-transformers for encoding text.

Produces L2-normalized 768-dim vectors (with BAAI/bge-base-en-v1.5).
The model is loaded lazily on first encode() call to keep import fast.

Adapted from MacroCounterFactual's TextEncoder (stripped of LoRA —
MCF validation showed frozen encoders are more robust).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

__all__ = ["TextEncoder"]


class TextEncoder:
    """Encode text into dense sentence embeddings.

    Parameters
    ----------
    model_name : HuggingFace sentence-transformers model id.
        'BAAI/bge-base-en-v1.5' -> 768-dim, strong retrieval quality.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading text model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Output vector dimension."""
        return self._load().get_sentence_embedding_dimension() or 768

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into L2-normalized embeddings.

        Parameters
        ----------
        texts : list of strings.

        Returns
        -------
        np.ndarray of shape (len(texts), dimension), float32, L2-normalized.
        """
        model = self._load()
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
