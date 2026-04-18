"""
Embedding Engine — generates dense vectors for semantic search.

Used by the Elasticsearch memory backend. Lazy-loads the model on first
use so startup is fast even if embeddings aren't immediately needed.

Default model: all-MiniLM-L6-v2 (384 dims, ~90MB, fast, high quality).
This is the same model ChromaDB uses internally, so quality is on par.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger("tamagi.embeddings")

_engine: "EmbeddingEngine | None" = None


class EmbeddingEngine:
    """Wraps sentence-transformers for dense vector generation.

    Instantiate once and reuse — the model is loaded into memory on
    first call to encode() and kept warm for the lifetime of the process.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model '%s' on %s ...", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            dim = self._model.get_sentence_embedding_dimension()
            logger.info("Embedding model ready (dim=%d).", dim)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )

    def encode(self, text: str) -> list[float]:
        """Return a normalised dense vector for a single string."""
        self._load()
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Return normalised dense vectors for a list of strings."""
        self._load()
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    @property
    def dim(self) -> int:
        """Embedding dimension (requires model to be loaded)."""
        self._load()
        return self._model.get_sentence_embedding_dimension()


def get_engine(model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> EmbeddingEngine:
    """Return the process-level singleton EmbeddingEngine."""
    global _engine
    if _engine is None or _engine.model_name != model_name or _engine.device != device:
        _engine = EmbeddingEngine(model_name, device=device)
    return _engine
