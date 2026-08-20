from pathlib import Path
import tomllib

from utils.chip import resolve_chip_runtime_manifest
from utils.path import get_buckyball_path


def bemu_manifest(chip: str, bbdir: str | None = None) -> Path:
  if not chip:
    raise ValueError("missing required parameter: chip")
  return resolve_chip_runtime_manifest(bbdir or get_buckyball_path(), chip, "bemu")


def bemu_core_manifest(chip: str, bbdir: str | None = None) -> Path:
  root = Path(bbdir or get_buckyball_path())
  chip_manifest = root / "examples" / "chips" / chip / "chip.toml"
  with chip_manifest.open("rb") as f:
    data = tomllib.load(f)
  runtime = data.get("runtime", {})
  rel = runtime.get("bemuCore") or runtime.get("bemu")
  manifest = (chip_manifest.parent / rel).resolve()
  if not manifest.is_file():
    raise ValueError(f"BEMU Core manifest does not exist: {manifest}")
  return manifest


def bemu_tile(chip: str, bbdir: str | None = None) -> Path | None:
  root = Path(bbdir or get_buckyball_path())
  chip_manifest = root / "examples" / "chips" / chip / "chip.toml"
  with chip_manifest.open("rb") as f:
    data = tomllib.load(f)
  rel = data.get("runtime", {}).get("bemuTile")
  return (chip_manifest.parent / rel).resolve() if rel else None
