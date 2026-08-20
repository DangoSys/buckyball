from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class CorePackage:
    name: str
    root: Path
    config: Path
    compiler: Path | None


def _load_toml(path: Path) -> dict:
    with path.open("rb") as source:
        data = tomllib.load(source)
    if not isinstance(data, dict):
        raise ValueError(f"TOML root must be a table: {path}")
    return data


def resolve_chip_runtime_manifest(bbdir: str, chip: str, runtime_name: str) -> Path:
    """Resolve a Cargo manifest declared by a chip's ``[runtime]`` table."""
    chip_root = Path(bbdir) / "examples" / "chips" / chip
    chip_manifest = chip_root / "chip.toml"
    if not chip_manifest.is_file():
        raise ValueError(f"Chip manifest does not exist: {chip_manifest}")

    data = _load_toml(chip_manifest)
    chip_table = data.get("chip")
    if not isinstance(chip_table, dict) or chip_table.get("name") != chip:
        raise ValueError(f"Chip manifest name mismatch: {chip_manifest}")

    runtime = data.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"Chip manifest missing [runtime]: {chip_manifest}")
    manifest_rel = runtime.get(runtime_name)
    if not isinstance(manifest_rel, str) or not manifest_rel:
        raise ValueError(f"Chip manifest missing runtime.{runtime_name}: {chip_manifest}")

    manifest = (chip_root / manifest_rel).resolve()
    if not manifest.is_file():
        raise ValueError(f"Chip runtime manifest does not exist: {manifest}")
    return manifest


def _core_table(path: Path) -> dict:
    core = _load_toml(path).get("core")
    if not isinstance(core, dict):
        raise ValueError(f"missing [core] table: {path}")
    return core


def available_cores(bbdir: str) -> list[str]:
    roots = Path(bbdir) / "examples" / "cores"
    if not roots.is_dir():
        return []
    return sorted(
        path.name
        for path in roots.iterdir()
        if (path / "manifest.toml").is_file()
    )


def resolve_core(bbdir: str, name: str, require_compiler: bool = False) -> CorePackage:
    root = Path(bbdir) / "examples" / "cores" / name
    manifest = root / "manifest.toml"
    if not manifest.is_file():
        raise ValueError(f"Core does not exist: {name}")

    core = _core_table(manifest)
    if core.get("name") != name:
        raise ValueError(f"Core manifest name mismatch: {manifest}")

    config_rel = core.get("config")
    if not isinstance(config_rel, str) or not config_rel:
        raise ValueError(f"Core manifest missing core.config: {manifest}")
    config = (root / config_rel).resolve()
    if not config.is_file():
        raise ValueError(f"Core configuration does not exist: {config}")

    compiler_rel = core.get("compiler")
    compiler = (root / compiler_rel).resolve() if isinstance(compiler_rel, str) else None
    if compiler is not None and not (compiler / "CMakeLists.txt").is_file():
        raise ValueError(f"Core compiler does not exist: {compiler}")
    if require_compiler and compiler is None:
        raise ValueError(f"Core has no compiler package: {name}")

    return CorePackage(name=name, root=root, config=config, compiler=compiler)


def available_compiler_chips(bbdir: str) -> list[str]:
    chips = Path(bbdir) / "examples" / "chips"
    if not chips.is_dir():
        return []

    result = []
    for root in chips.iterdir():
        manifest = root / "chip.toml"
        if not manifest.is_file():
            continue
        try:
            core_name = _load_toml(manifest).get("chip", {}).get("compilerCore")
            if isinstance(core_name, str):
                resolve_core(bbdir, core_name, require_compiler=True)
                result.append(root.name)
        except (ValueError, tomllib.TOMLDecodeError):
            continue
    return sorted(result)


def resolve_chip_compiler_core(bbdir: str, chip: str) -> CorePackage:
    manifest = Path(bbdir) / "examples" / "chips" / chip / "chip.toml"
    if not manifest.is_file():
        raise ValueError(f"Chip manifest does not exist: {manifest}")

    chip_table = _load_toml(manifest).get("chip")
    if not isinstance(chip_table, dict) or chip_table.get("name") != chip:
        raise ValueError(f"Chip manifest name mismatch: {manifest}")
    core_name = chip_table.get("compilerCore")
    if not isinstance(core_name, str) or not core_name:
        raise ValueError(f"Chip manifest missing chip.compilerCore: {manifest}")
    return resolve_core(bbdir, core_name, require_compiler=True)
