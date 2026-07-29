from pathlib import Path

from utils.path import get_buckyball_path


def bemu_manifest(chip: str, bbdir: str | None = None) -> Path:
  if not chip:
    raise ValueError("missing required parameter: chip")
  root = Path(bbdir or get_buckyball_path())
  manifest = root / "examples" / "chips" / chip / "emu" / "Cargo.toml"
  if not manifest.is_file():
    raise ValueError(f"bemu chip manifest does not exist: {manifest}")
  return manifest
