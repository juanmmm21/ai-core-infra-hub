"""Operations: inference client, tracing, evaluation, synthetic data, dataset versions, labeling.

Install with ``pip install "aicore[ops]"``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from .._deps import component
from ..errors import InferenceError, MissingExtraError
from ..types import Generation
from .rag_eval import BenchmarkResult, RAGBenchmark, load_examples, load_hotpotqa
from .router_loop import RoundReport, RouterImprovementLoop

__all__ = [
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
]

AREA = "ops"


def _httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise MissingExtraError(AREA, "httpx", exc) from exc
    return httpx


class InferenceClient:
    """Client for any OpenAI-compatible server: llm-inference-server, Ollama (``/v1``),
    llama.cpp's ``llama-server`` or a hosted API. Implements :class:`aicore.rag.Generator`."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "default",
        api_key: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        httpx = _httpx()
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def models(self) -> list[str]:
        response = self._client.get("/models")
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", [])]

    def chat(
        self, messages: Sequence[Mapping[str, str]], max_tokens: int = 256, temperature: float = 0.0, **extra: Any
    ) -> Generation:
        body = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            **extra,
        }
        start = time.perf_counter()
        try:
            response = self._client.post("/chat/completions", json=body)
            response.raise_for_status()
        except Exception as exc:
            raise InferenceError(f"chat completion failed: {exc}") from exc
        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return Generation(
            text=choice["message"].get("content") or "",
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason") or "",
            latency_s=time.perf_counter() - start,
        )

    def complete(self, prompt: str, *, max_tokens: int = 256, temperature: float = 0.0) -> str:
        """Single-turn chat; the :class:`aicore.rag.Generator` protocol."""
        return self.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature).text

    def stream(
        self, messages: Sequence[Mapping[str, str]], max_tokens: int = 256, temperature: float = 0.0
    ) -> Iterator[str]:
        """Yields text deltas as the server streams them (SSE)."""
        body = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        with self._client.stream("POST", "/chat/completions", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
                if delta:
                    yield delta


class Tracer:
    """llm-observability-tracer: nested spans with inputs, outputs, errors and timings.

    ``with tracer.span("retrieve", inputs={...}) as outputs: ...``; the tree is available from
    :meth:`tree` and can be rendered as an HTML flame graph. With ``otlp_endpoint`` every span is
    also exported to OpenTelemetry (GenAI semantic conventions; prompts and completions only with
    ``capture_content=True``). :meth:`record` adds spans measured elsewhere, with their own times.
    """

    def __init__(
        self,
        tracer: Any = None,
        otlp_endpoint: str | None = None,
        service_name: str = "ai-core",
        capture_content: bool = False,
    ) -> None:
        mod = component(AREA, "llm_observability_tracer")
        self._tracer = tracer if tracer is not None else mod.GlobalTracer()
        if otlp_endpoint:
            self._tracer.enable_otel(endpoint=otlp_endpoint, service_name=service_name, capture_content=capture_content)

    def record(
        self,
        name: str,
        start: float,
        end: float,
        parent: Any = None,
        error: str | None = None,
        **metadata: Any,
    ) -> Any:
        """A finished span from ``start`` to ``end`` (epoch seconds) under ``parent`` (the value
        a previous ``record`` returned). Metadata keys follow the tracer's GenAI mapping
        (``operation``, ``provider``, ``model``, ``input_tokens``...)."""
        return self._tracer.record_span(name, start, end, metadata, parent, error)

    def forget(self, root: Any) -> None:
        """Drop a recorded tree once exported (long-running processes)."""
        self._tracer.record_tree([root])

    def flush(self) -> None:
        otel = getattr(self._tracer, "otel", None)
        if otel is not None:
            otel.flush()

    @property
    def native(self) -> Any:
        """The component's tracer, for features the SDK does not wrap yet."""
        return self._tracer

    @contextmanager
    def span(self, name: str, inputs: Mapping[str, Any] | None = None, **metadata: Any) -> Iterator[dict[str, Any]]:
        outputs: dict[str, Any] = {}
        self._tracer.start_span(name, inputs=dict(inputs or {}), metadata=metadata)
        try:
            yield outputs
        except BaseException as exc:
            self._tracer.end_span(outputs=outputs, error=exc if isinstance(exc, Exception) else None)
            raise
        else:
            self._tracer.end_span(outputs=outputs)

    def tree(self) -> list[dict[str, Any]]:
        return list(self._tracer.get_trace_tree())

    def clear(self) -> None:
        self._tracer.clear()

    def flamegraph_html(self, path: str | os.PathLike[str]) -> None:
        self._tracer.generate_html_flamegraph(str(path))


class Evaluator:
    """llm-eval-harness: lexical metrics per pair and dataset summaries with regression gates."""

    def __init__(self) -> None:
        self._mod = component(AREA, "llm_eval_harness")
        self._suite = self._mod.EvalHarnessSuite()

    def score(self, prediction: str, reference: str) -> dict[str, float]:
        summary = self._suite.run_evaluation([{"candidate": prediction, "reference": reference}])
        report = summary.individual_results[0]
        data = report.model_dump() if hasattr(report, "model_dump") else dict(report)
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}

    def evaluate(self, pairs: Sequence[tuple[str, str]]) -> dict[str, float]:
        summary = self._suite.run_evaluation([{"candidate": p, "reference": r} for p, r in pairs])
        data = summary.model_dump()
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


class SyntheticData:
    """synthetic-data-generator: instruction and preference (DPO) samples with quality filters."""

    def __init__(self, min_length: int = 20, max_length: int = 1000, dedup_threshold: float = 0.5) -> None:
        mod = component(AREA, "synthetic_data_generator")
        self._gen = mod.SyntheticDataGenerator(
            min_length=min_length, max_length=max_length, dedup_threshold=dedup_threshold
        )

    @property
    def native(self) -> Any:
        return self._gen

    def instructions(self, topics: Sequence[str], per_topic: int = 5) -> list[dict[str, str]]:
        return [s.model_dump() for s in self._gen.generate_instruction_dataset(list(topics), count_per_topic=per_topic)]

    def preferences(self, topics: Sequence[str], per_topic: int = 5) -> list[dict[str, str]]:
        return [s.model_dump() for s in self._gen.generate_dpo_dataset(list(topics), count_per_topic=per_topic)]


class DatasetRepo:
    """dataset-version-control: content-addressed commits of a dataset file, logs and diffs."""

    def __init__(self, store_dir: str | os.PathLike[str] = ".dvc_store") -> None:
        mod = component(AREA, "dataset_version_control")
        self._vcs = mod.DatasetVCS(store_dir=str(store_dir))
        self._vcs.init()

    @property
    def native(self) -> Any:
        return self._vcs

    def commit(self, path: str | os.PathLike[str], message: str) -> str:
        return str(self._vcs.commit(str(path), message))

    def checkout(self, commit: str, target: str | os.PathLike[str]) -> None:
        self._vcs.checkout(commit, str(target))

    def log(self) -> list[dict[str, Any]]:
        return list(self._vcs.log())

    def diff(self, a: str, b: str) -> dict[str, Any]:
        return dict(self._vcs.diff(a, b))


def annotation_app(tokenizer: Any = None) -> Any:
    """llm-annotation-studio's FastAPI app, with ``tokenizer`` (e.g. :class:`aicore.rag.BPETokenizer`)
    connected to its token counter."""
    studio = component(AREA, "llm_annotation_studio.main")
    studio.set_tokenizer(tokenizer)
    return studio.app
