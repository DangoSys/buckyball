#!/usr/bin/env python3
"""Derive computed chip parameters from step1 config.json."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from ball_normalize import normalize_ball_domain


def _die(msg: str) -> None:
    raise ValueError(msg)


def _factory(config: dict[str, Any], context: str) -> str:
    value = config["factory"]
    if not isinstance(value, str) or len(value.split(".")) < 2 or not all(
        part.isidentifier() for part in value.split(".")
    ):
        _die(f"{context}: factory must be a Scala companion FQCN")
    return value


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
    if isinstance(cores, list):
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
    """The built-in main tile is tile 0; configured mounts follow in list order."""
    for key in ("compute_tiles", "tileTemplate", "main_core"):
        if key in topo:
            _die(f"invalid topology field: {key}")
    main = topo["main_tile"]
    if not isinstance(main, dict):
        _die("main_tile must be a table")
    out = [dict(main, tile_id=0, kind="main")]
    groups = topo.get("tiles", [])
    if not isinstance(groups, list):
        _die("tiles must be an array of tables")
    for group in groups:
        count = group["count"]
        if type(count) is not int or count < 1:
            _die("tile count must be a positive int")
        base = {k: v for k, v in group.items() if k != "count"}
        for _ in range(count):
            out.append(dict(base, tile_id=len(out), kind="tile"))
    return out


def iter_cores(topo: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for tile in iter_topology_tiles(topo):
        yield from _tile_cores(tile)


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
    bd = normalize_ball_domain(core)
    if not isinstance(bd, dict):
        return 0
    ball_num = bd.get("ballNum")
    if ball_num is None:
        return 0
    if not isinstance(ball_num, int) or ball_num < 0:
        _die("balldomain.ballNum must be a non-negative int")
    return ball_num


def _n_tiles(topo: dict[str, Any]) -> int:
    if "top" in topo:
        _die("invalid topology field: top")
    return len(iter_topology_tiles(topo))


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
            "cpu_kind": core["cpu"]["kind"],
            "factory_class": _factory(core, pkg),
            "config_path": _repo_rel(repo, rel),
            "balldomain_base_dir": _balldomain_base_dir(repo, core, pkg),
            "bank_num": num,
            "bank_width": width,
            "bank_entries": entries,
            "ball_num": ball_num,
        }
        if ball_num > 0:
            bd = normalize_ball_domain(core)
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


def _validate_spm(config: dict[str, Any]) -> None:
    base, size, bits = (config[key] for key in ("base", "bytes", "dataBits"))
    if any(type(value) is not int for value in (base, size, bits)):
        _die("SPM base/bytes/dataBits must be integers")
    if bits < 64 or bits & (bits - 1):
        _die("SPM dataBits must be a power of two >= 64")
    beat = bits // 8
    if size < 2 * beat or size & (size - 1) or base < 0 or base % beat or base + size > 1 << 64:
        _die("SPM requires an aligned base and power-of-two capacity within the 64-bit address space")


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
        controller = tile.get("controllerCore")
        if controller is not None and (type(controller) is not int or controller not in range(n)):
            _die(f"tile_id={tile_id}: controllerCore must name a local core")
        if tile["kind"] == "main" and controller is not None:
            _die("main tile has no task controller")
        ants = [core for core in tile_cores if core["cpu"]["kind"] == "ant"]
        if ants:
            shared_spm = tile.get("tss")
            if not isinstance(shared_spm, dict):
                _die("Ant compute tile requires explicit tss geometry")
            _validate_spm(shared_spm)
            for core in ants:
                params = core["cpu"]["config"]
                tls = params["tls"]
                _validate_spm(tls)
                code = params["codeBytes"]
                if type(code) is not int or code < 32 or code & (code - 1):
                    _die("Ant codeBytes must be a power of two >= 32")
                if not 1 <= params["taskBits"] <= 64:
                    _die("Ant taskBits must be in [1,64]")
                if min(tls["base"], shared_spm["base"]) < code:
                    _die("Ant code overlaps TLS/TSS")
                if not (tls["base"] + tls["bytes"] <= shared_spm["base"] or
                        shared_spm["base"] + shared_spm["bytes"] <= tls["base"]):
                    _die("Ant TLS/TSS windows overlap")
                if tls["dataBits"] != shared_spm["dataBits"]:
                    _die("Ant TLS/TSS access widths must match")
        shared = tile.get("sharedMem")
        if isinstance(shared, dict):
            if shared["bankWidth"] != 128:
                _die("sharedMem.bankWidth must be 128 bits")
            rows = shared["bankEntries"]
            if type(rows) is not int or rows < 2 or rows > 65536 or rows & (rows - 1):
                _die("sharedMem.bankEntries must be a power of two in [2,65536]")
            if shared["enable"] and (shared["entries"] <= 0 or shared["entries"] % rows):
                _die("sharedMem.entries must be a positive multiple of bankEntries")
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
        if tile["kind"] == "main":
            if "factory" in tile:
                _die("main tile must not specify a factory")
            factory = ""
        else:
            factory = _factory(tile, str(tile_path))
        placements.append(
            {
                "tile_id": tile_id,
                "kind": tile["kind"],
                "factory_class": factory,
                "path": _repo_rel(repo, tile_path),
                "core_indices": indices,
                "cores_per_tile": n,
                "virtual_bank_count": vbc,
                "mem_ball_channel_num": mem_ball_channel_num,
            }
        )
        if controller is not None:
            placements[-1]["controller_core_index"] = indices[controller]
        offset += n
    return placements


def _bemu_balls(repo: Path, topo: dict[str, Any]) -> list[dict[str, str]]:
    seen: set[str] = set()
    balls: list[dict[str, str]] = []
    for core in iter_cores(topo):
        bd = normalize_ball_domain(core)
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
            if funct7 == 0:
                _die(f"core {pkg}: funct7 zero is not a Buckyball instruction")
            if funct7 in {1, 16, 32, 33, 34, 35}:
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
    """The chip's own bebop-chip entry (empty for the default tile runner) and the tile it runs."""
    tiles = iter_topology_tiles(topo)
    mounted = [t["tile_id"] for t in tiles if t["kind"] == "tile"]
    main = repo / "examples" / "chips" / chip / "emu" / "src" / "main.rs"
    entry = f"examples/chips/{chip}/emu/src/main.rs" if main.is_file() else ""
    return entry, mounted[0] if mounted else 0


def _ball_ctest_dirs(repo: Path, core: dict[str, Any]) -> list[str]:
    bd = normalize_ball_domain(core)
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


def _derive_harts(tiles: list[dict[str, Any]], cores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only system CPUs receive hart IDs; Ant contexts are tile-local resources."""
    harts: list[dict[str, Any]] = []
    for tile in tiles:
        context = 0
        for local, core_index in enumerate(tile["core_indices"]):
            inst = cores[core_index]
            if inst["core_id"] != local:
                _die(f"tile {tile['tile_id']}: core IDs must match ordered slots")
            if inst["cpu_kind"] == "ant":
                inst["ant_context_id"] = context
                context += 1
                continue
            harts.append({
                "hart_id": len(harts), "tile_id": tile["tile_id"],
                "core_id": local, "core_index": core_index, "visible": True,
                "target": _target_name(inst["role"], inst["pkg"]),
                "pkg": inst["pkg"], "role": inst["role"],
            })
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

    cores = _derive_cores(repo, topo)
    tiles = _derive_tiles(repo, topo, cores)
    targets = _derive_targets(repo, topo, cores)
    harts = _derive_harts(tiles, cores)
    n_tiles = _n_tiles(topo)

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
