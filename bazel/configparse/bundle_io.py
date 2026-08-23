#!/usr/bin/env python3
"""Load ChipBundle protobuf written by chip_bundle.py / Bazel :bundle."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from chip_bundle_pb2 import ChipBundle  # type: ignore


def load_bundle(path: Path) -> ChipBundle:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing ChipBundle: {path}")
    bundle = ChipBundle()
    bundle.ParseFromString(path.read_bytes())
    if not bundle.name:
        raise ValueError(f"ChipBundle missing name: {path}")
    return bundle


def bundle_path(repo: Path, chip: str) -> Path:
    gen = repo / "examples" / "chips" / chip / "generated" / "chip.pb"
    if gen.is_file():
        return gen
    raise FileNotFoundError(
        f"missing {gen}; run: python3 bazel/configparse/chip_bundle.py --repo . --chip {chip} --all"
    )
