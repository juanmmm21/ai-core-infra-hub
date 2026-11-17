"""Text encoders behind one small protocol, so every component gets the same vectors.

Components each expected their own encoder (a hash, a sibling's checkpoint, a sentence
transformer); the SDK passes one :class:`Embedder` to all of them through the adapters.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .errors import AicoreError, MissingExtraError

_WORD = re.compile(r"\w+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into fixed-size vectors."""

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _strip_accents(text: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


class HashEmbedder:
    """Feature hashing of words and character trigrams into a signed, L2-normalized vector.

    No model and no download: lexical similarity only (shared words and spellings), stable
    across processes and machines. It is the default so that everything works offline and in
    CI; anything that needs meaning should pass a real model (:class:`SentenceTransformerEmbedder`
    or :class:`OllamaEmbedder`).
    """

    def __init__(self, dimension: int = 512, trigram_weight: float = 0.5) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._trigram_weight = trigram_weight

    @property
    def dimension(self) -> int:
        return self._dimension

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = _WORD.findall(_strip_accents(text.lower()))
        feats: list[tuple[str, float]] = [("w:" + w, 1.0) for w in words]
        for w in words:
            padded = f"#{w}#"
            feats.extend(("c:" + padded[i : i + 3], self._trigram_weight) for i in range(len(padded) - 2))
        return feats

    def embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self._dimension, dtype=np.float64)
        for feat, weight in self._features(text):
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self._dimension] += sign * weight
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return [float(x) for x in vec]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class SentenceTransformerEmbedder:
    """A ``sentence-transformers`` model (e.g. ``all-MiniLM-L6-v2``), loaded on first use."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None) -> None:
        self.model_name = model_name
        self._device = device
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise MissingExtraError("rag", "sentence_transformers", exc) from exc
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return [[float(x) for x in row] for row in vectors]


class OllamaEmbedder:
    """Embeddings from a local Ollama server (``/api/embed``), e.g. ``qwen3-embedding:0.6b``.

    Uses only the standard library, so it adds no dependency to whoever embeds the SDK.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 30.0,
        keep_alive: str = "5m",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed(["dimension probe"])[0])
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        body = json.dumps({"model": self.model, "input": list(texts), "keep_alive": self.keep_alive})
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read())
        except OSError as exc:
            raise AicoreError(f"Ollama embeddings failed: {exc}") from exc
        vectors = payload.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise AicoreError("Ollama returned no embeddings")
        out: list[list[float]] = []
        for row in vectors:
            norm = math.sqrt(sum(float(x) * float(x) for x in row)) or 1.0
            out.append([float(x) / norm for x in row])
        self._dimension = len(out[0])
        return out


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors (0.0 if either is zero)."""
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return float(va @ vb / (na * nb))
