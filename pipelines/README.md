# Pipelines

## Router improvement loop (`improve_router.py`)

The closed loop for an intent router, built from five components through `aicore.ops.RouterImprovementLoop`:

```
synthetic-data-generator ─► dataset-version-control ─► training ─► llm-eval-harness ─► semantic-model-router
  labelled messages from      train / eval splits        hashed      scorecards on the    registry: promote the
  specs/assistant_router.json committed, hashes kept     or LoRA/    same held-out        challenger only if it
                              with every model           QLoRA       templates            beats the champion
```

* **Evaluation never leaks.** Whole *templates* (phrasings) are held out for evaluation with a split
  seed fixed across rounds, so no model ever trains on an evaluation phrasing — not even a champion
  from an earlier round. (The first version split per round, and a weak champion "won" because it
  had trained on the next round's evaluation templates; the fixed split is the fix.)
* **Model selection never looks at the evaluation set.** Each candidate trains on 80 % of the
  training templates and early-stops on the other 20 %.
* **Promotion rule** (semantic-model-router `decide`): macro-F1 above the champion's, no class
  losing more than 0.02 recall, p95 latency under 50 ms. Every decision is appended to
  `registry/promotions.jsonl` with both scorecards.

### Documented run: `runs/2026-11-17/`

`python pipelines/improve_router.py --out pipelines/runs/2026-11-17` on a 4-vCPU Xeon container,
Python 3.13, torch 2.15 CPU. Full tables in [`runs/2026-11-17/REPORT.md`](runs/2026-11-17/REPORT.md);
models, promotion log and versioned datasets next to it.

| Round | Data (train / eval) | Champion before (macro-F1) | Challengers (macro-F1) | Decision |
| --- | --- | --- | --- | --- |
| r1-baseline | 128 / 52 | — | hashed v1 0.808 | v1 champion (first model) |
| r2-improve | 635 / 265 | v1 0.897 | hashed v2 0.923; LoRA/QLoRA MiniLM v3 0.835 | none: v2 loses 0.021 recall on `general` and 0.024 on `trivial`; v3 is worse |
| r3-more-data | 1,703 / 807 | v1 0.765 | hashed v4 0.811 | none: v4 loses 0.088 recall on `project` |

What the run shows: the loop works end to end and the gate does what it is for — two challengers
with a *better average* were rejected because each made one route worse (sending code work to the
general engine is exactly the costly mistake for an assistant). The LoRA-tuned multilingual
encoder (QLoRA: NF4-frozen base, rank-8 adapters on query/value, trained in 30 s on CPU) did not
beat a hashed linear model on these synthetic phrasings and costs ~20 ms per message instead of
~0.2 ms. The champion that serves is still v1. Next steps worth trying: more varied grammars (the
evaluation templates are hard for every lexical model), and the champion/challenger shadow mode on
real traffic before any promotion.
