#!/usr/bin/env python3
"""Emit cmake -D flags for workload ninja from ChipBundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bundle_io import bundle_path, load_bundle


def _die(msg: str) -> None:
    print(f"workload_cmake_defs: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--chip", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    chip = args.chip
    bundle = load_bundle(bundle_path(repo, chip))
    if bundle.name != chip:
        _die(f"bundle name {bundle.name!r} != chip {chip!r}")

    primary = bundle.workload.primary_core
    if not primary:
        _die(f"{chip} has no unique compiler core for chip-level workloads")

    for key, value in bundle.workload.cmake_defs.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
