"""Ops translations: evaluation reports into router scorecards, model kinds into the registry."""

from __future__ import annotations

from typing import Any

from .._deps import component


def scorecard_from_report(version: str, report: Any) -> Any:
    """llm-eval-harness ``ClassificationReport`` -> semantic-model-router ``Scorecard``.

    The two components describe a classifier's evaluation with the same numbers but their own
    types; the improvement loop evaluates with the harness and decides with the router."""
    router = component("agents", "semantic_model_router.registry")
    return router.Scorecard(
        version=version,
        n=int(report.n),
        accuracy=float(report.accuracy),
        macro_f1=float(report.macro_f1),
        per_class={k: dict(v) for k, v in report.per_class.items()},
        latency_ms_p50=float(report.latency_ms_p50),
        latency_ms_p95=float(report.latency_ms_p95),
        confusion={k: dict(v) for k, v in report.confusion.items()},
    )


def register_router_kinds() -> None:
    """Let the router's registry load LoRA classifiers trained by llm-qlora-finetuner.

    The router never imports the finetuner; this registration is the only link, made here."""
    router = component("agents", "semantic_model_router.registry")

    def load_lora(directory: str) -> Any:
        finetuner = component("train", "llm_qlora_finetuner.classifier")
        return finetuner.LoraClassifier.load(directory)

    router.register_kind("lora-classifier", load_lora)
