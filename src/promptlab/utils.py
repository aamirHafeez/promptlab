"""Small shared helpers."""

from __future__ import annotations

import re
from typing import Any

_VAR_PATTERN = re.compile(r"{{\s*(\w+)\s*}}")


def render(template: str, variables: dict[str, Any]) -> str:
    """Replace {{ var }} placeholders with values. Missing vars are left as-is."""

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(variables[key]) if key in variables else match.group(0)

    return _VAR_PATTERN.sub(_sub, template)


def template_vars(template: str) -> list[str]:
    """Return the variable names referenced in a template."""
    return sorted(set(_VAR_PATTERN.findall(template)))


def truncate(text: str, length: int = 60) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= length else text[: length - 1] + "\u2026"


def fmt_cost(cost: float) -> str:
    return "$0.00" if cost == 0 else f"${cost:.4f}"
