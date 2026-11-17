# Router improvement loop — documented run

Run on 2026-11-17 in 117 s · x86_64, 4 CPUs · Python 3.13.18 · grammar `assistant_router.json`.

Evaluation set: synthetic messages from **templates held out of every training set** (fixed across rounds).
Model selection: each candidate trains on 80 % of the training templates and keeps the epoch with the best accuracy on the other 20 % (early stopping); evaluation templates are never seen before the final scoring.
Promotion rule: macro-F1 must beat the champion's, no class may lose more than 0.02 recall, p95 latency per message under 50 ms.

## r1-baseline

Data: 128 training (of which 26 for early stopping) / 52 evaluation messages · dataset-version-control commits train `c79c0cb4c120`, eval `dfcdab6fe3c6`.

| Model | Version | Accuracy | Macro-F1 | Recall project | Recall general | Recall trivial | p50 ms | p95 ms | Promoted |
|---|---|---|---|---|---|---|---|---|---|
| hashed | v1 | 0.808 | 0.808 | 0.667 | 0.846 | 0.944 | 0.16 | 0.36 | yes |

* `hashed` (v1, trained in 0.1 s): first model: champion by default

Champion after the round: **v1**.

## r2-improve

Data: 635 training (of which 100 for early stopping) / 265 evaluation messages · dataset-version-control commits train `0c47272333b3`, eval `bbc32a358906`.

| Model | Version | Accuracy | Macro-F1 | Recall project | Recall general | Recall trivial | p50 ms | p95 ms | Promoted |
|---|---|---|---|---|---|---|---|---|---|
| champion (before) | v1 | 0.898 | 0.897 | 0.791 | 0.990 | 0.904 | 0.18 | 0.39 | — |
| hashed | v2 | 0.925 | 0.923 | 0.919 | 0.969 | 0.880 | 0.16 | 0.26 | no |
| lora-qlora-minilm | v3 | 0.834 | 0.835 | 0.756 | 0.760 | 1.000 | 20.81 | 30.11 | no |

* `hashed` (v2, trained in 1.1 s): recall of 'general' drops 0.021 (> 0.02); recall of 'trivial' drops 0.024 (> 0.02)
* `lora-qlora-minilm` (v3, trained in 30.5 s): macro-F1 0.8348 does not beat 0.8971 by more than 0.0; recall of 'general' drops 0.229 (> 0.02); recall of 'project' drops 0.035 (> 0.02)

Champion after the round: **v1**.

## r3-more-data

Data: 1703 training (of which 369 for early stopping) / 807 evaluation messages · dataset-version-control commits train `ab2da3158de4`, eval `1c6b3d66c115`.

| Model | Version | Accuracy | Macro-F1 | Recall project | Recall general | Recall trivial | p50 ms | p95 ms | Promoted |
|---|---|---|---|---|---|---|---|---|---|
| champion (before) | v1 | 0.765 | 0.765 | 0.663 | 0.819 | 0.862 | 0.16 | 0.29 | — |
| hashed | v4 | 0.812 | 0.811 | 0.575 | 0.972 | 1.000 | 0.14 | 0.23 | no |

* `hashed` (v4, trained in 4.2 s): recall of 'project' drops 0.088 (> 0.02)

Champion after the round: **v1**.

