import os
import tomllib
from pathlib import Path

from utils.path import get_buckyball_path


def available_bemu_chips(bbdir: str | None = None) -> list[str]:
    root = Path(bbdir or get_buckyball_path())
    cargo_toml = root / "bebop" / "Cargo.toml"
    with open(cargo_toml, "rb") as f:
        manifest = tomllib.load(f)

    features = manifest.get("features", {})
    chips = []
    for name in features:
        if not name.startswith("bemu-"):
            continue
        chip = name.removeprefix("bemu-")
        if (root / "examples" / "chips" / chip / "emu" / "Cargo.toml").is_file():
            chips.append(chip)
    return sorted(chips)


def bemu_feature(chip: str, bbdir: str | None = None) -> str:
    chips = available_bemu_chips(bbdir)
    if chip not in chips:
        available = ", ".join(chips) if chips else "none"
        raise ValueError(f"invalid chip: {chip}; available bemu chips: {available}")
    return f"bemu-{chip}"


def validate_bemu_chip(chip: str | None, bbdir: str | None = None) -> str:
    if not chip:
        raise ValueError("missing required parameter: chip")
    if not isinstance(chip, str):
        raise ValueError("chip must be a string")
    if os.path.sep in chip or chip in {".", ".."}:
        raise ValueError(f"invalid chip: {chip}")
    bemu_feature(chip, bbdir)
    return chip
