"""Training: LoRA / QLoRA adapters (llm-qlora-finetuner) on PyTorch modules.

Install with ``pip install "aicore[train]"`` (pulls PyTorch).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from .._deps import component

__all__ = ["LoraFinetuner"]

AREA = "train"


class LoraFinetuner:
    """Wraps ``QLoraPipeline``: swaps the target linear layers of a model for LoRA layers whose
    frozen base weight is optionally stored in NF4 (4 bits), so only the adapters train."""

    def __init__(
        self, r: int = 8, alpha: int = 16, target_modules: Sequence[str] = ("q_proj", "v_proj"), dropout: float = 0.05
    ) -> None:
        self._mod = component(AREA, "llm_qlora_finetuner")
        config = self._mod.LoraConfig(r=r, alpha=alpha, target_modules=list(target_modules), dropout=dropout)
        self._pipeline = self._mod.QLoraPipeline(config)

    @property
    def native(self) -> Any:
        return self._pipeline

    def apply(self, model: Any, quantize_4bit: bool = True) -> Any:
        return self._pipeline.apply_lora(model, quantize_4bit=quantize_4bit)

    def trainable_parameters(self, model: Any) -> list[tuple[str, Any]]:
        return list(self._pipeline.get_trainable_parameters(model))

    def save_adapters(self, model: Any, directory: str | os.PathLike[str]) -> str:
        return str(self._pipeline.save_adapters(model, str(directory)))

    def load_adapters(self, model: Any, directory: str | os.PathLike[str]) -> None:
        self._pipeline.load_adapters(model, str(directory))
