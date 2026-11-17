"""Errors raised by the SDK. Component exceptions are translated into these at the boundary."""

from __future__ import annotations


class AicoreError(Exception):
    """Base class for every error the SDK raises on purpose."""


class MissingExtraError(AicoreError, ImportError):
    """A component of an area is not installed: ``pip install "aicore[<area>]"``."""

    def __init__(self, area: str, module: str, cause: BaseException | None = None) -> None:
        self.area = area
        self.module = module
        super().__init__(
            f"'{module}' is not installed; it comes with the '{area}' extra: "
            f'pip install "aicore[{area}]"' + (f" ({cause})" if cause else "")
        )


class InferenceError(AicoreError):
    """An inference backend failed or answered something unusable."""
