"""The closed improvement loop: synthetic data -> DVC -> training -> harness -> promotion."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aicore.agents import IntentRouter
from aicore.ops import RouterImprovementLoop

SPEC = Path(__file__).resolve().parents[1] / "pipelines" / "specs" / "assistant_router.json"


def test_two_rounds_promote_only_a_winner(tmp_path: Path) -> None:
    loop = RouterImprovementLoop(str(SPEC), str(tmp_path))
    # Round 1: a deliberately weak model (few examples, few steps) becomes champion by default.
    r1 = loop.run("r1", n_per_label=20, seed=1, candidates={"weak": {"kind": "hashed", "fit": {"epochs": 1}}})
    assert r1.champion_before is None and r1.candidates[0].promoted and r1.champion_after == "v1"
    # Round 2: a proper challenger beats it on the same held-out templates; a copy of the weak
    # recipe does not.
    r2 = loop.run(
        "r2",
        n_per_label=150,
        seed=2,
        candidates={"strong": {"kind": "hashed"}, "weak-again": {"kind": "hashed", "fit": {"epochs": 1}}},
        max_class_drop=0.1,
    )
    strong, weak = r2.candidates
    assert strong.promoted, strong.reasons
    assert not weak.promoted
    assert r2.champion_after == strong.version
    assert strong.report["macro_f1"] > r2.champion_before_report["macro_f1"]
    # Data is versioned and every model records the commits it was trained on.
    log = loop.repo.log()
    assert len(log) == 4
    info = loop.registry.info(strong.version)
    assert info["data"] == r2.data_commits
    assert json.loads((tmp_path / "report_r2.json").read_text())["champion_after"] == strong.version
    # The SDK router serves the champion.
    router = IntentRouter(registry_dir=tmp_path / "registry")
    decision = router.route("arregla el test que falla en Atlas")
    assert decision.label == "project" and decision.path == f"champion:{strong.version}"
    assert router.route("pon una alarma a las ocho").label == "general"


@pytest.mark.skipif(not os.environ.get("AICORE_SLOW_TESTS"), reason="trains a transformer; set AICORE_SLOW_TESTS=1")
def test_lora_candidate(tmp_path: Path) -> None:
    pytest.importorskip("llm_qlora_finetuner")
    loop = RouterImprovementLoop(str(SPEC), str(tmp_path))
    r = loop.run(
        "lora",
        n_per_label=20,
        seed=3,
        candidates={
            "lora": {
                "kind": "lora",
                "init": {"base_model": "hf-internal-testing/tiny-random-bert"},
                "fit": {"epochs": 1},
            }
        },
    )
    assert r.candidates[0].version == "v1"
    assert IntentRouter(registry_dir=tmp_path / "registry").route("hola").label in {"project", "general", "trivial"}


def test_intent_router_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        IntentRouter()
