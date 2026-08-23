#!/usr/bin/env python3
"""Query flattened config JSON from toml2json. Only toml2json.py reads TOML."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

from toml2json import toml2json

_MILL = frozenset({"verilatorConfig", "p2eConfig"})
_FORBIDDEN = frozenset({"name", "topology", "compilerCore", "bemuTileIndex"})


def _die(msg: str) -> None:
    print(f"chip_json: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _walk(obj: Any) -> Iterator[Any]:
    if isinstance(obj, dict):
        yield obj
        for key, value in obj.items():
            if key != "includes":
                yield from _walk(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)


def core_pkg(rel: str) -> str | None:
    parts = Path(rel).parts
    if (
        len(parts) == 5
        and parts[0] == "examples"
        and parts[1] == "cores"
        and parts[3] == "configs"
        and parts[4].endswith(".toml")
    ):
        return parts[2]
    return None


def topology_path(repo: Path, chip: str) -> Path:
    path = repo / "examples" / "chips" / chip / "configs" / f"{chip}.toml"
    if not path.is_file():
        _die(f"missing topology {path}")
    return path


def load_topology(repo: Path, chip: str) -> dict[str, Any]:
    return toml2json(topology_path(repo, chip), repo.resolve())


def unique_cores(data: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for obj in _walk(data):
        if not isinstance(obj, dict):
            continue
        rel = obj.get("_file")
        if not isinstance(rel, str):
            continue
        pkg = core_pkg(rel)
        if pkg and pkg not in seen:
            seen.add(pkg)
            out.append(pkg)
    if not out:
        _die("topology has no cores/<package>/configs/*.toml include")
    return out


def compiler_core(data: dict[str, Any]) -> str:
    cores = unique_cores(data)
    return cores[0] if len(cores) == 1 else ""


def tile_files(data: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    tiles: list[str] = []
    for obj in _walk(data):
        if not isinstance(obj, dict):
            continue
        rel = obj.get("_file")
        if not isinstance(rel, str) or "/tiles/" not in rel or rel in seen:
            continue
        seen.add(rel)
        tiles.append(rel)
    tiles.sort()
    if not tiles:
        _die("topology has no tile file")
    return tiles


def _tile_cores(tile: dict[str, Any]) -> list[dict[str, Any]]:
    cores = tile.get("cores")
    if isinstance(cores, list) and cores:
        return cores
    template = tile.get("coreTemplate")
    if isinstance(template, dict):
        count = template.get("count")
        if not isinstance(count, int) or count < 1:
            _die("[coreTemplate].count must be a positive int")
        return [template] * count
    _die("tile must define [[cores]] or [coreTemplate]")


def iter_cores(data: dict[str, Any]) -> Iterator[dict[str, Any]]:
    tiles = data.get("tiles")
    if isinstance(tiles, list) and tiles:
        for tile in tiles:
            if not isinstance(tile, dict):
                _die("tiles entry must be a table")
            yield from _tile_cores(tile)
        return
    template = data.get("tileTemplate")
    if isinstance(template, dict):
        count = template.get("count")
        if not isinstance(count, int) or count < 1:
            _die("[tileTemplate].count must be a positive int")
        for _ in range(count):
            yield from _tile_cores(template)
        return
    _die("topology must define [[tiles]] or [tileTemplate]")


def core_entry(data: dict[str, Any], core_name: str) -> dict[str, Any]:
    matches = []
    for core in iter_cores(data):
        rel = core.get("_file")
        if isinstance(rel, str) and core_pkg(rel) == core_name:
            matches.append(core)
    if not matches:
        _die(f"topology core {core_name!r} not found")
    files = {c["_file"] for c in matches if isinstance(c.get("_file"), str)}
    if len(files) > 1:
        _die(f"topology has multiple configs for {core_name}")
    return matches[0]


def ball_dirs(data: dict[str, Any]) -> list[str]:
    mappings = data.get("ballIdMappings")
    if not isinstance(mappings, list):
        _die("ballIdMappings must be a list")
    dirs: list[str] = []
    seen: set[str] = set()
    for mapping in mappings:
        if not isinstance(mapping, dict):
            _die("ballIdMappings entry must be a table")
        cls = mapping.get("ballClass")
        if not isinstance(cls, str) or not cls.startswith("examples.balls."):
            _die(f"malformed ballClass: {cls!r}")
        directory = cls[len("examples.balls.") :].split(".", 1)[0]
        if not directory:
            _die(f"malformed ballClass: {cls!r}")
        if directory in seen:
            continue
        seen.add(directory)
        dirs.append(directory)
    return sorted(dirs)


def bank_params(core: dict, core_name: str, repo: Path) -> tuple[int, int, int]:
    mem = core.get("memdomain")
    if isinstance(mem, dict):
        bank = mem.get("bank")
    else:
        stub = (
            repo
            / "examples"
            / "cores"
            / core_name
            / "configs"
            / "memdomains"
            / "default.toml"
        )
        if not stub.is_file():
            _die(f"{core_name}: missing {stub}")
        bank = toml2json(stub, repo).get("bank")
    if not isinstance(bank, dict):
        _die(f"{core_name}: missing [bank]")
    num, width, entries = bank.get("num"), bank.get("width"), bank.get("entries")
    if not isinstance(num, int) or num <= 0:
        _die(f"{core_name}: bank.num must be a positive int")
    if not isinstance(width, int) or width < 8 or width % 8 != 0:
        _die(f"{core_name}: bank.width must be a positive multiple of 8")
    if not isinstance(entries, int) or entries <= 0:
        _die(f"{core_name}: bank.entries must be a positive int")
    return num, width, entries


def ball_ctest_dirs(repo: Path, core: dict) -> list[str]:
    balldomain = core.get("balldomain")
    if not isinstance(balldomain, dict):
        return []
    dirs = []
    for directory in ball_dirs(balldomain):
        path = repo / "examples" / "balls" / directory / "workloads" / "ctests"
        if not path.is_dir():
            _die(f"missing {path}")
        dirs.append(str(path))
    return dirs


def poly_placement(data: dict[str, Any]) -> dict[str, str]:
    ids: dict[str, list[str]] = {}
    for index, core in enumerate(iter_cores(data)):
        name = core.get("name")
        if not isinstance(name, str) or not name:
            _die("poly core missing name")
        ids.setdefault(name, []).append(str(index))
    out: dict[str, str] = {"BUCKYBALL_RUSHB_PLACEMENT_STRICT": "ON"}
    for name, vals in ids.items():
        out[f"_POLY_CORE_IDS_{name}"] = ";".join(vals)
    return out


def bemu_paths(repo: Path, chip: str, data: dict[str, Any]) -> tuple[str, int]:
    main = repo / "examples" / "chips" / chip / "emu" / "src" / "main.rs"
    tiles = tile_files(data)
    if main.is_file():
        if len(tiles) != 1:
            _die(f"chip {chip}: emu requires exactly one tile file, got {tiles}")
        return f"examples/chips/{chip}/emu/src/main.rs", 0
    return "", 0


def check_manifest(manifest: dict[str, Any], chip: str) -> None:
    if "runtime" in manifest:
        _die(f"{chip}: [runtime] is removed; use [chip].verilatorConfig / p2eConfig")
    for key in manifest:
        if key not in ("_file", "includes", "chip"):
            _die(f"{chip}: chip.toml unexpected key {key}")
    table = manifest.get("chip")
    if not isinstance(table, dict):
        _die(f"{chip}: missing [chip]")
    for key, value in table.items():
        if key in _FORBIDDEN:
            _die(f"{chip}: {key} is derived; do not set it in chip.toml")
        if key not in _MILL:
            _die(f"{chip}: unexpected [chip].{key}")
        if not isinstance(value, str) or not value:
            _die(f"{chip}: [chip].{key} must be a non-empty string")


def chip_index(repo: Path, chip: str) -> dict[str, Any]:
    from chip_bundle import chip_index as bundle_index

    return bundle_index(repo, chip)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--chip")
    parser.add_argument("--balldomain")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not repo.is_dir():
        _die(f"repo is not a directory: {repo}")

    if args.chip:
        out = chip_index(repo, args.chip)
    elif args.balldomain:
        data = toml2json(repo / args.balldomain, repo)
        out = {"ballDirs": ball_dirs(data), "includes": data.get("includes", [])}
    else:
        _die("pass --chip or --balldomain")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
