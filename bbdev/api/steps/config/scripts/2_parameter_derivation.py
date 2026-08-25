#!/usr/bin/env python3
"""Derive computed chip parameters from step1 config.json."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_SYSTEM_FUNCTS = frozenset({0, 1, 16, 32, 33, 34, 35})


def _die(msg: str) -> None:
    raise ValueError(msg)


def _repo_rel(repo: Path, path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    return p.resolve().relative_to(repo.resolve()).as_posix()


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
    if "cores" not in parts:
        return None
    idx = parts.index("cores")
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def _require_id(obj: dict[str, Any], key: str, ctx: str) -> int:
    val = obj.get(key)
    if not isinstance(val, int) or val < 0:
        _die(f"{ctx}: missing {key} (non-negative int required)")
    return val


def _require_id_list(obj: dict[str, Any], key: str, count: int, ctx: str) -> list[int]:
    val = obj.get(key)
    if not isinstance(val, list) or len(val) != count:
        _die(f"{ctx}: {key} must be a list of length {count}")
    out: list[int] = []
    for item in val:
        if not isinstance(item, int) or item < 0:
            _die(f"{ctx}: {key} entries must be non-negative ints")
        out.append(item)
    if len(set(out)) != len(out):
        _die(f"{ctx}: {key} has duplicates: {out}")
    return out


def _tile_cores(tile: dict[str, Any]) -> list[dict[str, Any]]:
    cores = tile.get("cores")
    if isinstance(cores, list) and cores:
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for i, core in enumerate(cores):
            if not isinstance(core, dict):
                _die("cores entry must be a table")
            core_id = _require_id(core, "core_id", f"cores[{i}]")
            if core_id in seen:
                _die(f"duplicate core_id {core_id} in tile")
            seen.add(core_id)
            out.append(core)
        expected = set(range(len(out)))
        if seen != expected:
            _die(f"tile core_id must be exactly 0..{len(out)-1}, got {sorted(seen)}")
        out.sort(key=lambda c: c["core_id"])
        return out

    template = tile.get("coreTemplate")
    if isinstance(template, dict):
        count = template.get("count")
        if not isinstance(count, int) or count < 1:
            _die("[coreTemplate].count must be a positive int")
        core_ids = _require_id_list(template, "core_ids", count, "coreTemplate")
        expected = set(range(count))
        if set(core_ids) != expected:
            _die(f"coreTemplate.core_ids must be exactly 0..{count-1}, got {core_ids}")
        base = {
            k: v
            for k, v in template.items()
            if k not in ("count", "core_ids")
        }
        out = []
        for core_id in sorted(core_ids):
            core = dict(base)
            core["core_id"] = core_id
            out.append(core)
        return out
    _die("tile must define [[cores]] or [coreTemplate]")


def iter_topology_tiles(topo: dict[str, Any]) -> list[dict[str, Any]]:
    tiles = topo.get("tiles")
    if isinstance(tiles, list) and tiles:
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        for i, tile in enumerate(tiles):
            if not isinstance(tile, dict):
                _die("tiles entry must be a table")
            tile_id = _require_id(tile, "tile_id", f"tiles[{i}]")
            if tile_id in seen:
                _die(f"duplicate tile_id {tile_id}")
            seen.add(tile_id)
            out.append(tile)
        expected = set(range(len(out)))
        if seen != expected:
            _die(f"tile_id must be exactly 0..{len(out)-1}, got {sorted(seen)}")
        out.sort(key=lambda t: t["tile_id"])
        return out

    template = topo.get("tileTemplate")
    if isinstance(template, dict):
        count = template.get("count")
        if not isinstance(count, int) or count < 1:
            _die("[tileTemplate].count must be a positive int")
        tile_ids = _require_id_list(template, "tile_ids", count, "tileTemplate")
        expected = set(range(count))
        if set(tile_ids) != expected:
            _die(f"tileTemplate.tile_ids must be exactly 0..{count-1}, got {tile_ids}")
        base = {
            k: v
            for k, v in template.items()
            if k not in ("count", "tile_ids")
        }
        out = []
        for tile_id in sorted(tile_ids):
            tile = dict(base)
            tile["tile_id"] = tile_id
            out.append(tile)
        return out
    _die("topology must define [[tiles]] or [tileTemplate]")


def iter_cores(topo: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for tile in iter_topology_tiles(topo):
        yield from _tile_cores(tile)


def _tile_signature(tile: dict[str, Any]) -> tuple[tuple[int, str, str], ...]:
    sig = []
    for core in _tile_cores(tile):
        rel = core.get("_file")
        if not isinstance(rel, str):
            _die("core config missing _file")
        pkg = core_pkg(rel)
        if not pkg:
            _die(f"unsupported core config path: {rel}")
        role = core.get("name")
        if role is not None and not isinstance(role, str):
            _die("core name must be a string")
        sig.append((core["core_id"], pkg, role or ""))
    return tuple(sig)


def _assert_isomorphic_tiles(topo: dict[str, Any]) -> None:
    tiles = iter_topology_tiles(topo)
    if not tiles:
        _die("no tiles")
    ref = _tile_signature(tiles[0])
    if not ref:
        _die("tile has no cores")
    for tile in tiles[1:]:
        sig = _tile_signature(tile)
        if sig != ref:
            _die(
                f"tiles must be isomorphic; tile_id={tile['tile_id']} "
                f"signature {sig} != tile_id={tiles[0]['tile_id']} {ref}"
            )


def hart_id(tile_id: int, core_id: int, cores_per_tile: int) -> int:
    if core_id < 0 or core_id >= cores_per_tile:
        _die(f"core_id {core_id} out of range for cores_per_tile={cores_per_tile}")
    return tile_id * cores_per_tile + core_id


def unique_cores(topo: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for obj in _walk(topo):
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


def tile_files(topo: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    tiles: list[str] = []
    for obj in _walk(topo):
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


def bank_params(core: dict, pkg: str) -> tuple[int, int, int]:
    mem = core.get("memdomain")
    if not isinstance(mem, dict):
        _die(f"{pkg}: missing memdomain")
    bank = mem.get("bank")
    if not isinstance(bank, dict):
        _die(f"{pkg}: missing [bank]")
    num, width, entries = bank.get("num"), bank.get("width"), bank.get("entries")
    if not isinstance(num, int) or num <= 0:
        _die(f"{pkg}: bank.num must be a positive int")
    if not isinstance(width, int) or width < 8 or width % 8 != 0:
        _die(f"{pkg}: bank.width must be a positive multiple of 8")
    if not isinstance(entries, int) or entries <= 0:
        _die(f"{pkg}: bank.entries must be a positive int")
    return num, width, entries


def _ball_dir(ball_class: str) -> str:
    if not ball_class.startswith("examples.balls."):
        _die(f"malformed ballClass: {ball_class!r}")
    directory = ball_class[len("examples.balls.") :].split(".", 1)[0]
    if not directory:
        _die(f"malformed ballClass: {ball_class!r}")
    return directory


def _balldomain_base_dir(repo: Path, core: dict, pkg: str) -> str:
    bd = core.get("balldomain")
    if isinstance(bd, dict):
        rel = bd.get("_file")
        if isinstance(rel, str):
            return _repo_rel(repo, Path(rel).parent)
    return f"examples/cores/{pkg}/configs/balldomains"


def _config_path(repo: Path, config: object) -> str:
    if isinstance(config, str):
        return _repo_rel(repo, config)
    if isinstance(config, dict):
        raw = config.get("_file")
        if isinstance(raw, str):
            return _repo_rel(repo, raw)
    _die(f"ball mapping config must be a path or include _file: {config!r}")


def _ball_num(core: dict) -> int:
    bd = core.get("balldomain")
    if not isinstance(bd, dict):
        return 0
    ball_num = bd.get("ballNum")
    if ball_num is None:
        return 0
    if not isinstance(ball_num, int) or ball_num < 0:
        _die("balldomain.ballNum must be a non-negative int")
    return ball_num


def _n_tiles(topo: dict[str, Any]) -> int:
    top = topo.get("top")
    if not isinstance(top, dict):
        _die("topology missing [top]")
    n = top.get("nTiles")
    if not isinstance(n, int) or n < 1:
        _die("[top].nTiles must be a positive int")
    return n


def _derive_cores(repo: Path, topo: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, core in enumerate(iter_cores(topo)):
        rel = core.get("_file")
        if not isinstance(rel, str):
            _die("core config missing _file")
        pkg = core_pkg(rel)
        if not pkg:
            _die(f"unsupported core config path: {rel}")
        role = core.get("name")
        if role is not None and not isinstance(role, str):
            _die("core name must be a string")
        core_id = _require_id(core, "core_id", f"core index {index}")
        ball_num = _ball_num(core)
        if isinstance(core.get("memdomain"), dict):
            num, width, entries = bank_params(core, pkg)
        elif ball_num > 0:
            _die(f"{pkg}: buckyball core missing memdomain")
        else:
            num, width, entries = 0, 0, 0
        entry: dict[str, Any] = {
            "index": index,
            "core_id": core_id,
            "pkg": pkg,
            "role": role or "",
            "config_path": _repo_rel(repo, rel),
            "balldomain_base_dir": _balldomain_base_dir(repo, core, pkg),
            "bank_num": num,
            "bank_width": width,
            "bank_entries": entries,
            "ball_num": ball_num,
        }
        if ball_num > 0:
            bd = core.get("balldomain")
            if not isinstance(bd, dict):
                _die(f"{pkg}: missing balldomain")
            mappings = bd.get("ballIdMappings")
            if not isinstance(mappings, list):
                _die(f"{pkg}: missing ballIdMappings")
            entry["mappings"] = []
            for mapping in mappings:
                if not isinstance(mapping, dict):
                    _die("ballIdMappings entry must be a table")
                ball_class = mapping.get("ballClass")
                if not isinstance(ball_class, str) or not ball_class:
                    _die(f"ballClass must be a non-empty string: {mapping!r}")
                entry["mappings"].append(
                    {
                        "ball_id": mapping.get("ballId"),
                        "ball_dir": _ball_dir(ball_class),
                        "config_path": _config_path(repo, mapping.get("config")),
                    }
                )
        out.append(entry)
    return out


def _derive_tiles(
    repo: Path, topo: dict[str, Any], cores: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    placements: list[dict[str, Any]] = []
    offset = 0
    for tile in iter_topology_tiles(topo):
        tile_id = _require_id(tile, "tile_id", "tile")
        tile_path = tile.get("_file")
        if not isinstance(tile_path, str):
            _die("tile missing _file")
        tile_cores = _tile_cores(tile)
        n = len(tile_cores)
        shared = tile.get("sharedMem")
        vbc = 0
        if isinstance(shared, dict):
            raw = shared.get("virtualBankCount")
            if isinstance(raw, int) and raw > 0:
                vbc = raw
        if vbc == 0 and n > 0:
            bank_nums = [cores[i]["bank_num"] for i in range(offset, offset + n)]
            if any(bank_nums):
                vbc = max(bank_nums)
        indices = list(range(offset, offset + n))
        has_buckyball = any(cores[i]["ball_num"] > 0 for i in indices)
        mem_ball_channel_num = 0
        if has_buckyball:
            raw = tile.get("memBallChannelNum")
            if not isinstance(raw, int):
                _die("tile with Buckyball cores must define memBallChannelNum")
            mem_ball_channel_num = raw
        placements.append(
            {
                "tile_id": tile_id,
                "path": _repo_rel(repo, tile_path),
                "core_indices": indices,
                "cores_per_tile": n,
                "virtual_bank_count": vbc,
                "mem_ball_channel_num": mem_ball_channel_num,
            }
        )
        offset += n
    return placements


def _bemu_balls(repo: Path, topo: dict[str, Any]) -> list[dict[str, str]]:
    seen: set[str] = set()
    balls: list[dict[str, str]] = []
    for core in iter_cores(topo):
        bd = core.get("balldomain")
        if not isinstance(bd, dict):
            continue
        mappings = bd.get("ballIdMappings")
        isa = bd.get("ballISA")
        if not isinstance(mappings, list) or not isinstance(isa, list):
            continue
        bid_to_class = {
            m["ballId"]: m["ballClass"]
            for m in mappings
            if isinstance(m, dict)
            and isinstance(m.get("ballId"), int)
            and isinstance(m.get("ballClass"), str)
        }
        pkg = core_pkg(core.get("_file", "")) or "core"
        for entry in isa:
            if not isinstance(entry, dict):
                _die("ballISA entry must be a table")
            funct7 = entry.get("funct7")
            bid = entry.get("bid")
            if not isinstance(funct7, int) or not isinstance(bid, int):
                _die(f"ballISA entry must have funct7 and bid: {entry!r}")
            if funct7 in _SYSTEM_FUNCTS:
                continue
            ball_class = bid_to_class.get(bid)
            if not ball_class:
                _die(
                    f"core {pkg}: ballISA funct7 {funct7} "
                    f"references missing bid {bid}"
                )
            if ball_class in seen:
                continue
            seen.add(ball_class)
            ball_dir = _ball_dir(ball_class)
            emu_lib = repo / "examples" / "balls" / ball_dir / "emu" / "src" / "lib.rs"
            if not emu_lib.is_file():
                _die(f"missing BEMU ball source for {ball_class}: {emu_lib}")
            balls.append(
                {
                    "ball_class": ball_class,
                    "ball_dir": ball_dir,
                    "emu_lib": _repo_rel(repo, emu_lib),
                }
            )
    return balls


def _bemu_paths(repo: Path, chip: str, topo: dict[str, Any]) -> tuple[str, int]:
    main = repo / "examples" / "chips" / chip / "emu" / "src" / "main.rs"
    tiles = tile_files(topo)
    if main.is_file():
        if len(tiles) != 1:
            _die(f"chip {chip}: emu requires exactly one tile file, got {tiles}")
        return f"examples/chips/{chip}/emu/src/main.rs", 0
    return "", 0


def _ball_ctest_dirs(repo: Path, core: dict[str, Any]) -> list[str]:
    bd = core.get("balldomain")
    if not isinstance(bd, dict):
        _die("core missing balldomain for ctest dirs")
    mappings = bd.get("ballIdMappings")
    if not isinstance(mappings, list):
        _die("core missing ballIdMappings for ctest dirs")
    dirs: list[str] = []
    for mapping in mappings:
        if not isinstance(mapping, dict):
            _die("ballIdMappings entry must be a table")
        ball_class = mapping.get("ballClass")
        if not isinstance(ball_class, str):
            _die("ballIdMappings entry missing ballClass")
        ball_dir = _ball_dir(ball_class)
        path = repo / "examples" / "balls" / ball_dir / "workloads" / "ctests"
        if not path.is_dir():
            _die(f"missing {path}")
        dirs.append(_repo_rel(repo, path))
    return dirs


def _target_name(role: str, pkg: str) -> str:
    if role:
        return role
    return pkg


def _derive_targets(
    repo: Path, topo: dict[str, Any], cores: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    topo_by_pkg: dict[str, dict[str, Any]] = {}
    for core in iter_cores(topo):
        rel = core.get("_file")
        if not isinstance(rel, str):
            continue
        pkg = core_pkg(rel)
        if pkg and pkg not in topo_by_pkg:
            topo_by_pkg[pkg] = core

    for inst in cores:
        name = _target_name(inst["role"], inst["pkg"])
        if name in targets:
            continue
        pkg = inst["pkg"]
        topo_core = topo_by_pkg.get(pkg)
        if topo_core is None:
            _die(f"topology missing core package {pkg}")
        target: dict[str, Any] = {
            "pkg": pkg,
            "compiler": "compiler",
            "bank": {
                "num": inst["bank_num"],
                "width": inst["bank_width"],
                "entries": inst["bank_entries"],
            },
            "ball_ctest_dirs": (
                _ball_ctest_dirs(repo, topo_core) if inst["ball_num"] > 0 else []
            ),
        }
        targets[name] = target
    return targets


def _derive_harts(
    tiles: list[dict[str, Any]], cores: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not tiles:
        _die("no tiles")
    cores_per_tile = tiles[0]["cores_per_tile"]
    for tile in tiles[1:]:
        if tile["cores_per_tile"] != cores_per_tile:
            _die("tiles must be isomorphic (cores_per_tile mismatch)")

    harts: list[dict[str, Any]] = []
    for tile in tiles:
        tile_id = tile["tile_id"]
        for local, core_index in enumerate(tile["core_indices"]):
            inst = cores[core_index]
            core_id = inst["core_id"]
            if core_id != local:
                _die(
                    f"tile_id={tile_id}: core_id={core_id} != sorted slot {local}"
                )
            hid = hart_id(tile_id, core_id, cores_per_tile)
            harts.append(
                {
                    "hart_id": hid,
                    "tile_id": tile_id,
                    "core_id": core_id,
                    "core_index": core_index,
                    "target": _target_name(inst["role"], inst["pkg"]),
                    "pkg": inst["pkg"],
                    "role": inst["role"],
                }
            )
    harts.sort(key=lambda h: h["hart_id"])
    ids = [h["hart_id"] for h in harts]
    if ids != list(range(len(harts))):
        _die(f"hart_id must be exactly 0..{len(harts)-1}, got {ids}")
    if len(harts) != len(cores):
        _die(f"hart count {len(harts)} != core count {len(cores)}")
    return harts


def _workload(chip: str, targets: dict[str, dict[str, Any]], repo: Path) -> dict[str, Any]:
    defs: dict[str, str] = {
        "BUCKYBALL_WORKLOAD_CHIP": chip,
        "BUCKYBALL_CARGO_TARGET_DIR": str(repo / "bebop" / "target" / chip),
    }
    return {"cmake_param": defs}


def derive(data: dict[str, Any], repo: Path, chip: str) -> dict[str, Any]:
    repo = repo.resolve()
    topo = data.get("designs")
    if not isinstance(topo, dict):
        _die(f"{chip}: missing designs")

    root_file = data.get("_file")
    if not isinstance(root_file, str):
        _die(f"{chip}: missing root _file")
    design_file = topo.get("_file")
    if not isinstance(design_file, str):
        _die(f"{chip}: designs missing _file")

    includes_raw = data.get("includes")
    if not isinstance(includes_raw, list):
        _die(f"{chip}: includes must be a list")

    _assert_isomorphic_tiles(topo)
    cores = _derive_cores(repo, topo)
    tiles = _derive_tiles(repo, topo, cores)
    targets = _derive_targets(repo, topo, cores)
    harts = _derive_harts(tiles, cores)
    n_tiles = _n_tiles(topo)
    if len(tiles) != n_tiles:
        _die(f"{chip}: [top].nTiles={n_tiles} but got {len(tiles)} tile(s)")

    chip_main, tile_index = _bemu_paths(repo, chip, topo)

    return {
        "chip": chip,
        "name": Path(_repo_rel(repo, design_file)).stem,
        "chip_path": _repo_rel(repo, root_file),
        "tile_config_path": _repo_rel(repo, design_file),
        "includes": [_repo_rel(repo, p) for p in includes_raw],
        "n_tiles": n_tiles,
        "targets": targets,
        "harts": harts,
        "cores": cores,
        "tiles": tiles,
        "bemu": {
            "chip_main": chip_main,
            "tile_index": tile_index,
            "balls": _bemu_balls(repo, topo),
        },
        "workload": _workload(chip, targets, repo),
    }


def write_derived(data: dict[str, Any], repo: Path, chip: str, out: Path) -> dict[str, Any]:
    derived = derive(data, repo, chip)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return derived


def main() -> None:
    if len(sys.argv) not in (4, 5):
        _die(f"usage: {sys.argv[0]} REPO CHIP CONFIG.json [OUT.json]")
    repo = Path(sys.argv[1])
    chip = sys.argv[2]
    config = Path(sys.argv[3])
    out = (
        Path(sys.argv[4])
        if len(sys.argv) == 5
        else config.parent / "derived.json"
    )
    data = json.loads(config.read_text(encoding="utf-8"))
    derived = write_derived(data, repo, chip, out)
    sys.stdout.write(json.dumps(derived, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
