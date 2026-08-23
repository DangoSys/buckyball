from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tomllib


@dataclass(frozen=True)
class CorePackage:
    name: str
    root: Path
    compiler: Path | None
    config: Path | None = None


_SKIP_CORE_DIRS = frozenset()
_CHIP_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_MILL_KEYS = frozenset({"verilatorConfig", "p2eConfig"})
_DERIVED_KEYS = frozenset({"name", "topology", "compilerCore", "bemuTileIndex"})


def bebop_cargo(bbdir: str) -> Path:
    manifest = Path(bbdir) / "bebop" / "Cargo.toml"
    if not manifest.is_file():
        raise ValueError(f"bebop Cargo.toml does not exist: {manifest}")
    return manifest


def bebop_bemu_cargo(bbdir: str, chip: str) -> Path:
    manifest = Path(bbdir) / "examples" / "chips" / chip / "generated" / "bemu" / "Cargo.toml"
    if not manifest.is_file():
        raise ValueError(f"bebop-bemu manifest does not exist: {manifest}")
    return manifest


def require_chip(data: dict) -> str:
    chip = data.get("chip")
    if not isinstance(chip, str) or not chip:
        raise ValueError("Missing required parameter: --chip")
    extra = data.get("config")
    if isinstance(extra, str) and extra and extra != "None":
        raise ValueError(
            "do not pass --config; mill class comes from chip.toml [chip] "
            "verilatorConfig / p2eConfig"
        )
    if not _CHIP_RE.fullmatch(chip):
        raise ValueError(f"invalid chip: {chip}")
    return chip


def _chip_table(data: dict, chip: str) -> dict:
    if "runtime" in data:
        raise ValueError(
            f"{chip}: [runtime] is removed; mill class is [chip].verilatorConfig / p2eConfig"
        )
    extra = sorted(k for k in data if k != "chip")
    if extra:
        raise ValueError(f"{chip}: unexpected top-level keys in chip.toml: {extra}")
    table = data.get("chip")
    if not isinstance(table, dict):
        raise ValueError(f"{chip}: missing [chip] in chip.toml")
    for key, value in table.items():
        if key in _DERIVED_KEYS:
            raise ValueError(
                f"{chip}: {key} is derived from the chip directory / topology; do not set it"
            )
        if key not in _MILL_KEYS:
            raise ValueError(f"{chip}: unexpected [chip].{key} in chip.toml")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{chip}: [chip].{key} must be a non-empty string")
    return table


def chip_toml(bbdir: str, chip: str) -> dict:
    chip = require_chip({"chip": chip})
    path = Path(bbdir) / "examples" / "chips" / chip / "chip.toml"
    if not path.is_file():
        raise ValueError(f"missing chip.toml: {path}")
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"chip.toml root must be a table: {path}")
    _chip_table(data, chip)
    return data


def chip_field(bbdir: str, chip: str, key: str) -> str:
    if key not in _MILL_KEYS:
        raise ValueError(f"invalid chip field: {key}")
    value = chip_toml(bbdir, chip)["chip"].get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{chip}: missing [chip].{key} in chip.toml")
    if "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"{chip}: [chip].{key} is not a single path component: {value!r}")
    return value


def available_cores(bbdir: str) -> list[str]:
    roots = Path(bbdir) / "examples" / "cores"
    if not roots.is_dir():
        return []
    return sorted(
        path.name
        for path in roots.iterdir()
        if path.is_dir()
        and path.name not in _SKIP_CORE_DIRS
        and not path.name.startswith(".")
        and (path / "configs").is_dir()
    )


def resolve_core(bbdir: str, name: str, require_compiler: bool = False) -> CorePackage:
    root = Path(bbdir) / "examples" / "cores" / name
    if not root.is_dir() or name in _SKIP_CORE_DIRS:
        raise ValueError(f"Core does not exist: {name}")
    if not (root / "configs").is_dir():
        raise ValueError(f"Core missing configs/: {root}")

    compiler_dir = root / "compiler"
    compiler = compiler_dir if (compiler_dir / "CMakeLists.txt").is_file() else None
    if require_compiler and compiler is None:
        raise ValueError(f"Core has no compiler package: {name}")

    return CorePackage(name=name, root=root, compiler=compiler, config=None)


def available_compiler_chips(bbdir: str) -> list[str]:
    chips = Path(bbdir) / "examples" / "chips"
    if not chips.is_dir():
        return []
    return sorted(
        path.name
        for path in chips.iterdir()
        if path.is_dir() and (path / "chip.toml").is_file()
    )
