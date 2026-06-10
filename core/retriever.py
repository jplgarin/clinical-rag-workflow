"""Local, dependency-light retrieval over plain-text knowledge bases.

Embeddings are computed with sentence-transformers so the whole thing runs
offline. The vector store is a thin wrapper around a numpy matrix; for the
corpus sizes we expect (a few hundred chunks per domain) a brute-force cosine
search is plenty and saves us a heavyweight vector DB dependency.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np

from core.schema import RetrievedChunk

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".txt", ".md"}
# Roughly a paragraph. Tuned by eye on the bundled corpora, not sacred.
_CHUNK_TARGET_CHARS = 800


def _split_into_chunks(text: str, target_chars: int = _CHUNK_TARGET_CHARS) -> list[str]:
    """Split on blank lines, then greedily pack paragraphs up to a budget."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if buffer and len(buffer) + len(para) > target_chars:
            chunks.append(buffer.strip())
            buffer = para
        else:
            buffer = f"{buffer}\n\n{para}" if buffer else para
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


class _Document:
    __slots__ = ("chunk_id", "content", "source", "metadata")

    def __init__(self, chunk_id: str, content: str, source: str, metadata: dict):
        self.chunk_id = chunk_id
        self.content = content
        self.source = source
        self.metadata = metadata


class VectorStore:
    """An in-memory embedded corpus for one knowledge domain.

    The embedding model is loaded lazily so that constructing a store (for
    example to immediately ``load_index``) does not pull sentence-transformers
    into memory unless we actually need to embed.
    """

    def __init__(self, name: str = "default", model_name: str = "all-MiniLM-L6-v2"):
        self.name = name
        self.model_name = model_name
        self._documents: list[_Document] = []
        self._embeddings: Optional[np.ndarray] = None
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # Imported here so test code can run without the heavy dependency
            # as long as it stubs out embedding.
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def __len__(self) -> int:
        return len(self._documents)

    def load_documents(self, path: Path) -> None:
        """Read every ``.txt``/``.md`` file under ``path`` and chunk it.

        Args:
            path: Directory to scan recursively. Missing directories raise so
                misconfiguration fails loudly rather than silently retrieving
                nothing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"knowledge path does not exist: {path}")

        files = sorted(
            p for p in path.rglob("*") if p.suffix.lower() in _SUPPORTED_SUFFIXES
        )
        for file in files:
            text = file.read_text(encoding="utf-8")
            for i, chunk in enumerate(_split_into_chunks(text)):
                self._documents.append(
                    _Document(
                        chunk_id=f"{file.stem}::{i}",
                        content=chunk,
                        source=file.name,
                        metadata={"path": str(file), "citation": file.stem},
                    )
                )
        logger.info(
            "loaded %d chunks from %d files in %s",
            len(self._documents),
            len(files),
            path,
        )

    def embed_documents(self) -> None:
        """Compute and L2-normalise embeddings for all loaded chunks."""
        if not self._documents:
            logger.warning("embed_documents called with no documents loaded")
            self._embeddings = np.empty((0, 0), dtype=np.float32)
            return
        contents = [d.content for d in self._documents]
        vectors = np.asarray(
            self.model.encode(contents, show_progress_bar=False),
            dtype=np.float32,
        )
        self._embeddings = _normalize(vectors)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks by cosine similarity."""
        if self._embeddings is None or self._embeddings.size == 0:
            return []
        query_vec = _normalize(
            np.asarray(self.model.encode([query]), dtype=np.float32)
        )
        scores = (self._embeddings @ query_vec.T).ravel()
        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [self._to_chunk(i, float(scores[i])) for i in top_idx]

    def save_index(self, path: Path) -> None:
        """Persist documents and embeddings to a single pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "model_name": self.model_name,
            "documents": [
                (d.chunk_id, d.content, d.source, d.metadata)
                for d in self._documents
            ],
            "embeddings": self._embeddings,
        }
        with path.open("wb") as fh:
            pickle.dump(payload, fh)
        logger.info("saved index '%s' to %s", self.name, path)

    def load_index(self, path: Path) -> None:
        """Restore a previously saved index."""
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        self.name = payload["name"]
        self.model_name = payload["model_name"]
        self._documents = [
            _Document(*doc) for doc in payload["documents"]
        ]
        self._embeddings = payload["embeddings"]
        logger.info("loaded index '%s' (%d chunks)", self.name, len(self._documents))

    def _to_chunk(self, index: int, score: float) -> RetrievedChunk:
        doc = self._documents[index]
        return RetrievedChunk(
            content=doc.content,
            source=doc.source,
            relevance_score=max(0.0, min(1.0, (score + 1.0) / 2.0)),
            chunk_id=doc.chunk_id,
            metadata={**doc.metadata, "store": self.name},
        )


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class Retriever:
    """Fan-out retrieval across one or more :class:`VectorStore` instances."""

    def __init__(self, stores: Optional[list[VectorStore]] = None):
        self.stores: list[VectorStore] = stores or []

    def add_store(self, store: VectorStore) -> None:
        self.stores.append(store)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Search every store, drop duplicates, and keep the best ``top_k``.

        Dedup is by ``chunk_id``; when the same chunk surfaces from more than
        one store we keep the higher-scoring hit.
        """
        best: dict[str, RetrievedChunk] = {}
        for store in self.stores:
            for chunk in store.search(query, top_k):
                existing = best.get(chunk.chunk_id)
                if existing is None or chunk.relevance_score > existing.relevance_score:
                    best[chunk.chunk_id] = chunk
        ranked = sorted(best.values(), key=lambda c: c.relevance_score, reverse=True)
        return ranked[:top_k]

    def retrieve_for_sections(
        self, sections: list[str], top_k: int = 5
    ) -> dict[str, list[RetrievedChunk]]:
        """Retrieve chunks keyed by report section.

        Each section title doubles as the query. That is deliberately simple;
        an adapter that wants richer per-section queries can compose them
        before calling in.
        """
        return {section: self.retrieve(section, top_k) for section in sections}
