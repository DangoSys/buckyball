import json
from pathlib import Path

from utils.build import install_bundle
from utils.chip import require_chip


def _repo(bbdir: str | None) -> Path:
    from utils.path import get_buckyball_path

    return Path(bbdir or get_buckyball_path())


def _chip_json(chip: str, bbdir: str | None = None) -> dict:
    root = _repo(bbdir)
    path = root / "examples" / "chips" / chip / "generated" / "chip.json"
    if not path.is_file():
        install_bundle(str(root), chip)
    if not path.is_file():
        raise ValueError(f"missing {path} after bundle build")
    return json.loads(path.read_text(encoding="utf-8"))


def bemu_manifest(chip: str, bbdir: str | None = None) -> Path:
    chip = require_chip({"chip": chip})
    root = _repo(bbdir)
    install_bundle(str(root), chip)
    manifest = root / "examples" / "chips" / chip / "generated" / "bemu" / "Cargo.toml"
    if not manifest.is_file():
        raise ValueError(f"missing {manifest}; bundle install failed")
    return manifest


def bemu_core_manifest(chip: str, bbdir: str | None = None) -> Path:
    return bemu_manifest(chip, bbdir)


def chip_emu_manifest(chip: str, bbdir: str | None = None) -> Path | None:
    chip = require_chip({"chip": chip})
    root = _repo(bbdir)
    main = _chip_json(chip, bbdir).get("bemu", {}).get("chipMain", "")
    if not main:
        return None
    manifest = root / "examples" / "chips" / chip / "emu" / "Cargo.toml"
    if not manifest.is_file():
        raise ValueError(f"missing chip emu manifest: {manifest}")
    return manifest


def bemu_tile_index(chip: str, bbdir: str | None = None) -> int | None:
    bemu = _chip_json(chip, bbdir).get("bemu", {})
    if not bemu.get("chipMain", ""):
        return None
    index = bemu.get("tileIndex", 0)
    if not isinstance(index, int) or index < 0:
        raise ValueError(f"chip {chip}: bemu.tileIndex must be a non-negative int")
    return index
