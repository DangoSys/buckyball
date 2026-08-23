#!/usr/bin/env python3
"""Generate bank_params.sh from JSON (memdomain bank section)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _die(msg: str) -> None:
    print(f"bank_params: {msg}", file=sys.stderr)
    raise SystemExit(1)


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

    bank = data.get("bank")
    if not isinstance(bank, dict):
        _die("JSON must contain bank object with num, width, and entries")

    num = bank.get("num")
    width = bank.get("width")
    entries = bank.get("entries")
    if num is None or width is None or entries is None:
        _die("memdomain [bank] must set num, width, and entries")
    if not isinstance(num, int) or num <= 0:
        _die("memdomain [bank].num must be a positive integer")
    if not isinstance(width, int) or width < 8 or width % 8 != 0:
        _die("memdomain [bank].width must be a positive multiple of 8 bits")
    if not isinstance(entries, int) or entries <= 0:
        _die("memdomain [bank].entries must be a positive integer")

    content = "\n".join(
        [
            f"BANK_NUM={num}",
            f"BANK_WIDTH_BITS={width}",
            f"BANK_WIDTH_BYTES={width // 8}",
            f"BANK_DEPTH={entries}",
            "",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
