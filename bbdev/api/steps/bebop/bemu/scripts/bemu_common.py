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


def bemu_core_manifest(chip: str, bbdir: str | None = None) -> Path:
    return bemu_manifest(chip, bbdir)


def chip_emu_manifest(chip: str, bbdir: str | None = None) -> Path | None:
    chip = require_chip({"chip": chip})
    root = _repo(bbdir)
    main = _chip(chip, bbdir).bemu.chip_main
    if not main:
        return None
    return root / "examples" / "chips" / chip / "emu" / "Cargo.toml"


def bemu_tile_index(chip: str, bbdir: str | None = None) -> int | None:
    bemu = _chip(chip, bbdir).bemu
    if not bemu.chip_main:
        return None
    return bemu.tile_index
