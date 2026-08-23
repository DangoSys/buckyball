#!/usr/bin/env python3
"""Generate ballISA.h from JSON (ballISA array or balldomain object)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _die(msg: str) -> None:
    print(f"ballISA: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _ball_isa(data: dict[str, Any]) -> list[Any]:
    if "ballISA" not in data:
        _die("JSON must contain ballISA array")
    isa = data["ballISA"]
    if not isinstance(isa, list):
        _die("ballISA must be an array")
    return isa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    if not args.json.is_file():
        _die(f"missing JSON: {args.json}")
    with args.json.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        _die("JSON root must be an object")

    isa = _ball_isa(data)
    seen_mne: dict[str, bool] = {}
    seen_f7: dict[int, bool] = {}
    entries: list[tuple[str, int]] = []

    for i, entry in enumerate(isa):
        if not isinstance(entry, dict):
            _die(f"ballISA[{i}] must be an object")
        mne = entry.get("mnemonic")
        if not isinstance(mne, str) or not mne:
            _die(f"ballISA[{i}]: mnemonic must be a non-empty string")
        f7 = entry.get("funct7")
        if not isinstance(f7, int) or f7 < 0:
            _die(f"ballISA[{i}]: funct7 must be a non-negative integer")
        if mne in seen_mne:
            _die(f"duplicate mnemonic {mne}")
        if f7 in seen_f7:
            _die(f"duplicate funct7 {f7}")
        seen_mne[mne] = True
        seen_f7[f7] = True
        entries.append((mne, f7))

    lines = [
        "/* This file is auto-generated. Do not edit. */",
        "#ifndef BB_BALL_ISA_H",
        "#define BB_BALL_ISA_H",
        "",
    ]
    for mne, f7 in entries:
        lines.append(f"#define BB_FUNC7_{mne} {f7}")
    lines.extend(["", "#endif", ""])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
