"""Lazy imports of the components, with an error that names the extra to install."""

from __future__ import annotations

import importlib
from types import ModuleType

from .errors import MissingExtraError


def component(area: str, module: str) -> ModuleType:
    """Import ``module`` (a component package) or explain which ``aicore[area]`` provides it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        top = module.split(".", 1)[0]
        if exc.name and exc.name.split(".", 1)[0] != top:
            raise  # a dependency *of* the component is broken: do not hide it
        raise MissingExtraError(area, module, exc) from exc
