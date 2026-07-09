import os
from pathlib import Path

from utils.path import get_buckyball_path


def validate_chip(chip: str | None) -> str:
  if not chip:
    raise ValueError("missing required parameter: chip")
  if not isinstance(chip, str):
    raise ValueError("chip must be a string")
  if os.path.sep in chip or chip in {".", ".."}:
    raise ValueError(f"invalid chip: {chip}")
  return chip


def chip_regression_dir(chip: str, backend: str, bbdir: str | None = None) -> Path:
  chip = validate_chip(chip)
  root = Path(bbdir or get_buckyball_path())
  return root / "examples" / "chips" / chip / "regression" / "batch" / backend


def regression_workload_toml(
  chip: str,
  backend: str,
  test_type: str,
  bbdir: str | None = None,
) -> str:
  if test_type == "elf-tests":
    suffix = "elf"
  elif test_type == "pk-tests":
    suffix = "pk"
  else:
    raise ValueError(f"invalid test type: {test_type}")

  toml = chip_regression_dir(chip, backend, bbdir) / f"workloads-{suffix}.toml"
  if not toml.is_file():
    raise ValueError(f"regression workload toml does not exist: {toml}")
  return str(toml)
