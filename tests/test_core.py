"""Core of the SDK: embedders, types, adapters' parity checks, and the frozen public surface."""

from __future__ import annotations

import math
from typing import ClassVar

import pytest

import aicore
from aicore import HashEmbedder, cosine
from aicore.adapters import check_fields
from aicore.adapters.agents import guard_category, verdict_from_shield
from aicore.adapters.rag import distance_to_score
from aicore.errors import AicoreError

# The stable API. Changing these lists is a deliberate API change: bump the minor version and
# document it in docs/sdk.md.
PUBLIC = {
    "aicore": [
        "AicoreError",
        "Answer",
        "Chunk",
        "Document",
        "Embedder",
        "Generation",
        "GuardVerdict",
        "HashEmbedder",
        "Hit",
        "InferenceError",
        "MemoryHit",
        "MissingExtraError",
        "OllamaEmbedder",
        "RouteDecision",
        "SentenceTransformerEmbedder",
        "ToolResult",
        "Triple",
        "__version__",
        "cosine",
    ],
    "aicore.rag": [
        "BPETokenizer",
        "Chunker",
        "DocumentParser",
        "Generator",
        "GraphExtractor",
        "HybridRetriever",
        "RAGPipeline",
        "Reranker",
        "VectorIndex",
    ],
    "aicore.agents": ["Guard", "IntentRouter", "Memory", "Orchestrator", "Router", "ToolRuntime"],
    "aicore.ops": [
        "BenchmarkResult",
        "DatasetRepo",
        "Evaluator",
        "InferenceClient",
        "RAGBenchmark",
        "RoundReport",
        "RouterImprovementLoop",
        "SyntheticData",
        "Tracer",
        "annotation_app",
        "load_examples",
        "load_hotpotqa",
    ],
    "aicore.train": ["LoraFinetuner"],
}


@pytest.mark.parametrize("module", sorted(PUBLIC))
def test_public_surface_is_frozen(module: str) -> None:
    import importlib

    mod = importlib.import_module(module)
    assert sorted(mod.__all__) == sorted(PUBLIC[module])
    for name in mod.__all__:
        assert hasattr(mod, name), f"{module}.{name} is exported but missing"


def test_hash_embedder_is_deterministic_normalized_and_lexical() -> None:
    emb = HashEmbedder(dimension=256)
    a, a2 = emb.embed(["the cat sat on the mat"])[0], emb.embed(["the cat sat on the mat"])[0]
    assert a == a2
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)
    near = emb.embed(["a cat sat on a mat"])[0]
    far = emb.embed(["quantum chromodynamics lattice"])[0]
    assert cosine(a, near) > cosine(a, far) + 0.3
    # accents do not matter
    assert cosine(emb.embed(["canción"])[0], emb.embed(["cancion"])[0]) > 0.99


def test_hash_embedder_rejects_bad_dimension() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dimension=0)


def test_cosine_of_zero_vector_is_zero() -> None:
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_distance_to_score_is_higher_is_better() -> None:
    assert distance_to_score(0.1, "cosine") > distance_to_score(0.5, "cosine")
    assert distance_to_score(1.0, "l2") > distance_to_score(3.0, "l2")
    assert distance_to_score(-5.0, "dot") > distance_to_score(-1.0, "dot")


def test_shield_reasons_map_to_stable_categories() -> None:
    assert guard_category("Entrada bloqueada por sospecha de Prompt Injection / Jailbreak.") == "prompt_injection"
    assert guard_category("Salida bloqueada debido a deteccion de fuga de informacion confidencial") == "secret_leak"
    assert verdict_from_shield((True, "hola [EMAIL]", "")).text == "hola [EMAIL]"
    blocked = verdict_from_shield((False, "", "lenguaje ofensivo"))
    assert (blocked.allowed, blocked.text, blocked.category) == (False, "", "toxicity")


def test_check_fields_detects_divergence() -> None:
    class Model:
        model_fields: ClassVar[dict[str, None]] = {"a": None}

    with pytest.raises(AicoreError, match="diverged"):
        check_fields(Model, ["a", "b"], "test")
    check_fields(Model, ["a"], "test")


def test_version() -> None:
    assert aicore.__version__.count(".") == 2
