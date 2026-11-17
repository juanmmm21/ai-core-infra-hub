"""aicore.ops over the real components; the inference client against a stub OpenAI server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from aicore.ops import DatasetRepo, Evaluator, InferenceClient, SyntheticData, Tracer, annotation_app
from aicore.rag import BPETokenizer


def _openai_stub(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/models"):
        return httpx.Response(200, json={"data": [{"id": "tiny"}]})
    body = json.loads(request.content)
    if body.get("stream"):
        chunks = [{"choices": [{"delta": {"content": t}}]} for t in ("Hola", " mundo")]
        text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
    return httpx.Response(
        200,
        json={
            "model": "tiny",
            "choices": [{"message": {"content": f"eco: {body['messages'][-1]['content']}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )


@pytest.fixture
def client() -> InferenceClient:
    c = InferenceClient("http://test/v1", model="tiny")
    c._client = httpx.Client(base_url="http://test/v1", transport=httpx.MockTransport(_openai_stub))
    return c


def test_inference_client_chat_complete_stream(client: InferenceClient) -> None:
    assert client.models() == ["tiny"]
    gen = client.chat([{"role": "user", "content": "hola"}])
    assert (gen.text, gen.prompt_tokens, gen.completion_tokens, gen.finish_reason) == ("eco: hola", 3, 2, "stop")
    assert client.complete("x") == "eco: x"
    assert list(client.stream([{"role": "user", "content": "x"}])) == ["Hola", " mundo"]


def test_tracer_nested_spans_and_errors() -> None:
    tracer = Tracer()
    tracer.clear()
    with tracer.span("pipeline", inputs={"q": "hola"}) as out:
        with tracer.span("retrieve") as inner:
            inner["hits"] = 3
        out["answer"] = "ok"
    with pytest.raises(RuntimeError), tracer.span("fails"):
        raise RuntimeError("boom")
    tree = tracer.tree()
    names = [n["name"] for n in tree]
    assert names[:2] == ["pipeline", "fails"]
    assert tree[0]["children"][0]["outputs"] == {"hits": 3}
    assert "boom" in str(tree[1]["error"])


def test_tracer_records_spans_with_times_and_exports(tmp_path: Path) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = Tracer()
    tracer.native.otel = __import__("llm_observability_tracer").OTelBridge(tracer_provider=provider)
    root = tracer.record("turn", 10.0, 12.0, origin="telegram")
    tracer.record("llm", 10.5, 11.9, parent=root, operation="chat", provider="ollama", model="m")
    names = sorted(s.name for s in exporter.get_finished_spans())
    assert names == ["chat m", "turn"]
    tracer.forget(root)
    assert tracer.tree() == []


def test_tracer_flamegraph(tmp_path: Path) -> None:
    tracer = Tracer()
    tracer.clear()
    with tracer.span("x"):
        pass
    tracer.flamegraph_html(tmp_path / "f.html")
    assert (tmp_path / "f.html").read_text(encoding="utf-8").count("traceData") >= 1


def test_evaluator_scores() -> None:
    ev = Evaluator()
    same = ev.score("Paris is the capital", "Paris is the capital")
    other = ev.score("Berlin", "Paris is the capital")
    assert same["exact_match"] == 1.0 and other["exact_match"] == 0.0
    assert ev.evaluate([("a b", "a b"), ("c", "d")])["total_samples"] == 2


def test_synthetic_data_and_dataset_repo(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = SyntheticData().instructions(["python", "sql"], per_topic=2)
    assert rows and {"instruction", "output"} <= set(rows[0])
    data = tmp_path / "train.json"
    data.write_text(json.dumps(rows), encoding="utf-8")
    repo = DatasetRepo(tmp_path / "store")
    first = repo.commit(data, "first")
    data.write_text(json.dumps([*rows, {"instruction": "nuevo", "output": "ejemplo"}]), encoding="utf-8")
    second = repo.commit(data, "second")
    assert [c["message"] for c in repo.log()][:2] in (["second", "first"], ["first", "second"])
    assert repo.diff(first, second)
    repo.checkout(first, tmp_path / "restored.json")
    assert json.loads((tmp_path / "restored.json").read_text()) == rows


def test_annotation_app_counts_with_injected_tokenizer() -> None:
    from fastapi.testclient import TestClient

    tok = BPETokenizer().train("hola mundo hola mundo", vocab_size=260)
    app = annotation_app(tok)
    data = TestClient(app).post("/api/tokenize", json={"text": "hola mundo"}).json()
    assert data["fallback_used"] is False and data["tokens_count"] == tok.count("hola mundo")
    annotation_app(None)
