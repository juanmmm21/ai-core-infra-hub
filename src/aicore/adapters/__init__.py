"""Translation between the components' own types and the SDK's (`aicore.types`).

This package is to the SDK what ``nanosql/translate.py`` is to Strata: the components never
import each other, so each keeps its own shapes (``{"distance": ...}`` dicts, pydantic models,
tuples) and these modules are the only place where they meet. Translations are mechanical;
they make no decisions of their own. Where a translation relies on a component's field names,
:func:`check_fields` verifies them once, so a renamed field fails loudly here and not as a
``KeyError`` in the middle of a request.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..errors import AicoreError

_checked: set[str] = set()


def check_fields(model: Any, expected: Iterable[str], where: str) -> None:
    """Fail if the pydantic ``model`` of a component no longer has the ``expected`` fields."""
    key = f"{getattr(model, '__module__', '')}.{getattr(model, '__name__', model)!s}"
    if key in _checked:
        return
    fields = set(getattr(model, "model_fields", {}) or {})
    missing = set(expected) - fields
    if missing:
        raise AicoreError(f"{where}: {key} has diverged, missing fields {sorted(missing)}")
    _checked.add(key)
