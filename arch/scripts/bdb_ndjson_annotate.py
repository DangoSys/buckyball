#!/usr/bin/env python3
"""
Lookup instruction name by funct id.

Example:
  python3 arch/scripts/bdb_ndjson_annotate.py 0x20
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

FILE_RE = re.compile(r"^(\d+)_([a-z0-9_]+)\.c$")


def parse_funct_id(v: Any) -> int:
    try:
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            return int(v, 0)
    except ValueError as e:
        raise SystemExit(f"invalid funct id: {v!r}") from e
    raise SystemExit(f"invalid funct id type: {type(v)}")


def build_funct_map(isa_dir: Path) -> dict[int, str]:
    if not isa_dir.is_dir():
        raise SystemExit(f"ISA dir not found: {isa_dir}")
    mp: dict[int, str] = {}
    for p in isa_dir.glob("*.c"):
        m = FILE_RE.match(p.name)
        if not m:
            continue
        fid = int(m.group(1))
        name = m.group(2)
        mp[fid] = name
    if not mp:
        raise SystemExit(f"no ISA funct mapping found under: {isa_dir}")
    return mp


def resolve_funct_name(fid: int, funct_map: dict[int, str]) -> str:
    if fid not in funct_map:
        raise SystemExit(f"unknown funct id: 0x{fid:x}")
    return funct_map[fid]


def default_isa_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "bb-tests"
        / "workloads"
        / "lib"
        / "bbhw"
        / "isa"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Lookup instruction name by funct id")
    p.add_argument("funct_id", help="funct id, e.g. 0x20 or 32")
    p.add_argument(
        "--isa-dir", type=Path, default=default_isa_dir(), help="ISA macro dir"
    )
    args = p.parse_args()

    fid = parse_funct_id(args.funct_id)
    fmap = build_funct_map(args.isa_dir)
    name = resolve_funct_name(fid, fmap)
    print(name)


if __name__ == "__main__":
    main()
