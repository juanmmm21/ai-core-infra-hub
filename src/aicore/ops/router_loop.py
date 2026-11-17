"""The closed improvement loop for an intent router.

    synthetic data ─► dataset-version-control ─► training ─► llm-eval-harness ─► promotion
    (generate_labeled)  (commit train & eval)    (hashed /     (same held-out      (champion /
                                                   LoRA-QLoRA)   eval set)          challenger)

Each round generates a labelled dataset from a grammar (synthetic-data-generator), splits it with
whole templates held out for evaluation, commits both files to a dataset repository
(dataset-version-control) so every model records the exact data it saw, trains one or more
challengers (``hashed``: the router's own NumPy classifier; ``lora``: llm-qlora-finetuner's
LoRA/QLoRA encoder classifier), evaluates champion and challengers on the **same** held-out set
with llm-eval-harness, and lets semantic-model-router's registry promote a challenger only if it
wins. Everything a round did is returned and written as JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .._deps import component
from ..adapters.ops import register_router_kinds, scorecard_from_report

AREA = "ops"


@dataclass
class CandidateResult:
    name: str
    version: str
    train_seconds: float
    report: dict[str, Any]
    promoted: bool
    reasons: list[str]


@dataclass
class RoundReport:
    round: str
    seed: int
    data_commits: dict[str, str]
    sizes: dict[str, int]
    champion_before: str | None
    champion_before_report: dict[str, Any] | None
    candidates: list[CandidateResult] = field(default_factory=list)
    champion_after: str | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record(sample: Any) -> dict[str, Any]:
    key = sample.text + "|" + ",".join(sample.context) + "|" + sample.label
    return {
        "id": hashlib.sha1(key.encode()).hexdigest()[:16],
        "text": sample.text,
        "label": sample.label,
        "context": list(sample.context),
        "template": sample.template,
    }


class RouterImprovementLoop:
    def __init__(self, spec_path: str, workdir: str, labels: list[str] | None = None) -> None:
        self.gen = component(AREA, "synthetic_data_generator.labeled")
        self.harness = component(AREA, "llm_eval_harness.classification")
        router = component("agents", "semantic_model_router")
        self.router = router
        dvc = component(AREA, "dataset_version_control")
        register_router_kinds()
        self.spec = self.gen.load_spec(spec_path)
        self.labels = labels or list(self.spec["labels"])
        self.workdir = workdir
        os.makedirs(os.path.join(workdir, "data"), exist_ok=True)
        self.registry = router.ModelRegistry(os.path.join(workdir, "registry"))
        self.repo = dvc.DatasetVCS(store_dir=os.path.join(workdir, "dvc_store"))
        self.repo.init()

    def _evaluate(self, model: Any, rows: list[dict[str, Any]]) -> Any:
        def predict(text: str, ctx: Any) -> str:
            p = model.predict_proba([text], [ctx])[0]
            best = max(range(len(model.labels)), key=lambda i: p[i])
            return str(model.labels[best])

        return self.harness.evaluate_classifier(
            predict, [r["text"] for r in rows], [r["label"] for r in rows], [r["context"] for r in rows]
        )

    def _train(
        self, kind: str, rows: list[dict[str, Any]], options: dict[str, Any], val_rows: list[dict[str, Any]] | None
    ) -> Any:
        texts = [r["text"] for r in rows]
        labels = [r["label"] for r in rows]
        contexts = [r["context"] for r in rows]
        fit = dict(options.get("fit", {}))
        if val_rows:
            fit["validation"] = (
                [r["text"] for r in val_rows],
                [r["label"] for r in val_rows],
                [r["context"] for r in val_rows],
            )
        if kind == "hashed":
            return self.router.IntentClassifier(self.labels).fit(texts, labels, contexts, **fit)
        if kind == "lora":
            finetuner = component("train", "llm_qlora_finetuner")
            clf = finetuner.LoraClassifier(self.labels, **options.get("init", {}))
            return clf.fit(texts, labels, contexts, **fit)
        raise ValueError(f"unknown candidate kind {kind!r}")

    def run(
        self,
        round_name: str,
        n_per_label: int = 300,
        seed: int = 0,
        eval_fraction: float = 0.3,
        split_seed: int = 0,
        candidates: dict[str, dict[str, Any]] | None = None,
        min_gain: float = 0.0,
        max_class_drop: float = 0.02,
        max_latency_ms: float | None = None,
    ) -> RoundReport:
        """One round. ``candidates`` maps a name to ``{"kind": "hashed"|"lora", "fit": {...},
        "init": {...}}``; by default a single hashed challenger. ``seed`` draws new samples;
        ``split_seed`` fixes which templates are evaluation-only, across rounds."""
        start = time.perf_counter()
        candidates = candidates or {"hashed": {"kind": "hashed"}}
        samples = self.gen.generate_labeled(self.spec, n_per_label, seed=seed)
        # The held-out *templates* must be the same in every round (``split_seed`` does not change
        # with ``seed``): otherwise a champion trained in an earlier round may have seen this
        # round's evaluation templates and win by memory, not by quality.
        train_s, eval_s = self.gen.split(samples, eval_fraction, seed=split_seed)
        train_rows, eval_rows = [_record(s) for s in train_s], [_record(s) for s in eval_s]
        # Early stopping looks at templates held out of *training* (never at the evaluation set).
        inner_s, val_s = self.gen.split(train_s, 0.2, seed=split_seed + 1)
        inner_rows, val_rows = [_record(s) for s in inner_s], [_record(s) for s in val_s]
        commits = {}
        for name, rows in (("train", train_rows), ("eval", eval_rows)):
            path = os.path.join(self.workdir, "data", f"router_{name}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, ensure_ascii=False)
            commits[name] = self.repo.commit(path, f"{round_name}: {name} split (seed {seed}, {len(rows)} rows)")

        champion = self.registry.champion_version()
        champ_report = self._evaluate(self.registry.load(champion), eval_rows) if champion else None
        report = RoundReport(
            round=round_name,
            seed=seed,
            data_commits=commits,
            sizes={"train": len(train_rows), "eval": len(eval_rows), "validation": len(val_rows)},
            champion_before=champion,
            champion_before_report=champ_report.to_dict() if champ_report else None,
        )
        champ_card = scorecard_from_report(champion, champ_report) if champion and champ_report else None
        for name, options in candidates.items():
            t0 = time.perf_counter()
            early = options.get("early_stopping", True)
            model = self._train(
                options["kind"], inner_rows if early else train_rows, options, val_rows if early else None
            )
            train_seconds = time.perf_counter() - t0
            version = self.registry.register(
                model, {"round": round_name, "candidate": name, "data": commits, "options": options.get("fit", {})}
            )
            ev = self._evaluate(model, eval_rows)
            card = scorecard_from_report(version, ev)
            if champ_card is None and self.registry.champion_version() == version:
                promoted, reasons = True, ["first model: champion by default"]
                with open(os.path.join(self.registry.root, "promotions.jsonl"), "a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps({"promoted": True, "champion": None, "challenger": asdict(card), "reasons": reasons})
                        + "\n"
                    )
            else:
                decision = self.registry.apply(version, champ_card, card, min_gain, max_class_drop, max_latency_ms)
                promoted, reasons = decision.promoted, list(decision.reasons)
            if promoted:
                champ_card = card
            report.candidates.append(CandidateResult(name, version, train_seconds, ev.to_dict(), promoted, reasons))
        report.champion_after = self.registry.champion_version()
        report.seconds = time.perf_counter() - start
        with open(os.path.join(self.workdir, f"report_{round_name}.json"), "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
        return report
