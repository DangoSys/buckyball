"""Small report helpers for dynamic-power-only summaries."""

from __future__ import annotations

import re
from pathlib import Path


_POWER_VALUE = re.compile(r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\s*([munp]?W)\b")
_UNIT_TO_WATTS = {"W": 1.0, "mW": 1e-3, "uW": 1e-6, "nW": 1e-9, "pW": 1e-12}


def _watts(value: str) -> float:
    match = _POWER_VALUE.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported power value: {value}")
    return float(match.group(1)) * _UNIT_TO_WATTS[match.group(2)]


def read_dynamic_power(report: str | Path) -> dict[str, str] | None:
    path = Path(report)
    if not path.is_file():
        return None
    values: dict[str, str] = {}
    patterns = {
        "internal_power": r"Cell Internal Power\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?\s*[munp]?W)\b",
        "switching_power": r"Net Switching Power\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?\s*[munp]?W)\b",
        "total_power": r"Total Power\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?\s*[munp]?W)\b",
    }
    text = path.read_text(encoding="utf-8", errors="ignore")
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            values[key] = match.group(1).strip()
    if "internal_power" not in values or "switching_power" not in values:
        return None
    dynamic_watts = _watts(values["internal_power"]) + _watts(values["switching_power"])
    values["dynamic_power"] = f"{dynamic_watts:.12g} W"
    return values
