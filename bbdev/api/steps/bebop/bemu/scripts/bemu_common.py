from pathlib import Path

from utils.event_common import require_chip


def _repo(bbdir: str | None) -> Path:
    from utils.path import get_buckyball_path

    return Path(bbdir or get_buckyball_path())


def bemu_manifest(chip: str, bbdir: str | None = None) -> Path:
    chip = require_chip({"chip": chip})
    root = _repo(bbdir)
    path = (
        root
        / "examples"
        / "chips"
        / chip
        / "configs"
        / "generated"
        / "bemu"
        / "Cargo.toml"
    )
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run bbdev config --install")
    return path


def bemu_chip_binary(chip: str) -> str:
    """The chip's BEMU boot entry in its generated crate."""
    return f"bebop-chip-{require_chip({'chip': chip})}"
