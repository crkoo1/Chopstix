"""Small helpers shared by the command-line tools."""

import json


def parse_overrides(items: list[str] | None) -> dict:
    """["key=value", ...] -> {key: value}; values are parsed as JSON when possible (numbers, lists, true/false)."""
    out = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out
