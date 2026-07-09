from pathlib import Path

from utils.path import get_buckyball_path


def bemu_chip_inst(chip: str, bbdir: str | None = None) -> Path:
  if not chip:
    raise ValueError("missing required parameter: chip")
  root = Path(bbdir or get_buckyball_path())
  inst = root / "examples" / "chips" / chip / "emu" / "src" / "lib.rs"
  if not inst.is_file():
    raise ValueError(f"bemu chip instruction file does not exist: {inst}")
  return inst


def bemu_env(chip: str, bbdir: str | None = None) -> dict[str, str]:
  return {"BEBOP_BEMU_CHIP_INST": str(bemu_chip_inst(chip, bbdir))}
