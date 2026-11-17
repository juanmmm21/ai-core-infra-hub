"""One documented run of the router improvement loop.

    python pipelines/improve_router.py --out pipelines/runs/2026-11-17

Round 1 (baseline): the router's hashed classifier trained on a small synthetic sample becomes
the first champion. Round 2: a larger sample, two challengers — the hashed classifier again and a
LoRA/QLoRA multilingual MiniLM from llm-qlora-finetuner. Round 3: more data again, the hashed
classifier. All evaluated on the same held-out templates; the registry promotes a challenger
only if it wins. Writes the round reports,
the registry (models, champion pointer, promotion log), the versioned datasets and REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from aicore.ops import RouterImprovementLoop

HERE = Path(__file__).resolve().parent
LABELS = ["project", "general", "trivial"]


def fmt_card(name: str, version: str, r: dict, promoted: object = "") -> str:
    pc = r["per_class"]
    return (
        f"| {name} | {version} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | "
        + " | ".join(f"{pc.get(c, {}).get('recall', 0.0):.3f}" for c in LABELS)
        + f" | {r['latency_ms_p50']:.2f} | {r['latency_ms_p95']:.2f} | {promoted} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--spec", default=str(HERE / "specs" / "assistant_router.json"))
    parser.add_argument("--base-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--lora-epochs", type=int, default=10, help="maximum; early stopping picks the epoch")
    parser.add_argument("--max-latency-ms", type=float, default=50.0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    loop = RouterImprovementLoop(args.spec, str(out))
    t0 = time.time()
    r1 = loop.run("r1-baseline", n_per_label=60, seed=1, candidates={"hashed": {"kind": "hashed"}})
    r2 = loop.run(
        "r2-improve",
        n_per_label=300,
        seed=2,
        candidates={
            "hashed": {"kind": "hashed", "fit": {"epochs": 100}},
            "lora-qlora-minilm": {
                "kind": "lora",
                "init": {"base_model": args.base_model, "quantize_4bit": True, "threads": os.cpu_count()},
                "fit": {"epochs": args.lora_epochs, "lr": 2e-3, "batch_size": 16, "patience": 2},
            },
        },
        max_latency_ms=args.max_latency_ms,
    )
    r3 = loop.run(
        "r3-more-data",
        n_per_label=900,
        seed=3,
        candidates={"hashed": {"kind": "hashed", "fit": {"epochs": 100}}},
        max_latency_ms=args.max_latency_ms,
    )
    lines = [
        "# Router improvement loop — documented run",
        "",
        f"Run on {time.strftime('%Y-%m-%d')} in {time.time() - t0:.0f} s · {platform.machine()}, "
        f"{os.cpu_count()} CPUs · Python {platform.python_version()} · grammar `{Path(args.spec).name}`.",
        "",
        "Evaluation set: synthetic messages from **templates held out of every training set** (fixed across rounds).",
        "Model selection: each candidate trains on 80 % of the training templates and keeps the epoch with the best "
        "accuracy on the other 20 % (early stopping); evaluation templates are never seen before the final scoring.",
        "Promotion rule: macro-F1 must beat the champion's, no class may lose more than 0.02 recall, "
        f"p95 latency per message under {args.max_latency_ms:.0f} ms.",
        "",
    ]
    for r in (r1, r2, r3):
        lines += [
            f"## {r.round}",
            "",
            f"Data: {r.sizes['train']} training (of which {r.sizes['validation']} for early stopping) / "
            f"{r.sizes['eval']} evaluation messages · "
            f"dataset-version-control commits train `{r.data_commits['train'][:12]}`, "
            f"eval `{r.data_commits['eval'][:12]}`.",
            "",
            "| Model | Version | Accuracy | Macro-F1 | Recall project | Recall general | Recall trivial "
            "| p50 ms | p95 ms | Promoted |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        if r.champion_before_report:
            lines.append(fmt_card("champion (before)", r.champion_before or "", r.champion_before_report, "—"))
        for c in r.candidates:
            lines.append(fmt_card(c.name, c.version, c.report, "yes" if c.promoted else "no"))
        lines += [""]
        for c in r.candidates:
            lines.append(f"* `{c.name}` ({c.version}, trained in {c.train_seconds:.1f} s): {'; '.join(c.reasons)}")
        lines += ["", f"Champion after the round: **{r.champion_after}**.", ""]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    with open(out / "rounds.json", "w", encoding="utf-8") as fh:
        json.dump({"r1": r1.to_dict(), "r2": r2.to_dict(), "r3": r3.to_dict()}, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
