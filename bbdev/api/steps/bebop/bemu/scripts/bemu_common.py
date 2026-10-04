import sys
from pathlib import Path

from utils.event_common import require_chip


def _repo(bbdir: str | None) -> Path:
    from utils.path import get_buckyball_path

    return Path(bbdir or get_buckyball_path())


def _chip(chip: str, bbdir: str | None = None):
    chip = require_chip({"chip": chip})
    root = _repo(bbdir)
    scripts = root / "bbdev" / "api" / "steps" / "config" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import chip_pb2

    path = root / "examples" / "chips" / chip / "configs" / "generated" / "chip.pb"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run bbdev config --install")
    msg = chip_pb2.Chip()
    msg.ParseFromString(path.read_bytes())
    return msg


def bemu_manifest(chip: str, bbdir: str | None = None) -> Path:
    chip = require_chip({"chip": chip})
    root = _repo(bbdir)
    path = root / "examples" / "chips" / chip / "configs" / "generated" / "bemu" / "Cargo.toml"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run bbdev config --install")
    return path


def bemu_chip_binary(chip: str) -> str:
    """The chip's tile-level BEMU entry in its generated crate."""
    return f"bebop-chip-{require_chip({'chip': chip})}"


def bemu_tile_index(chip: str, bbdir: str | None = None) -> int:
    return _chip(chip, bbdir).bemu.tile_index
