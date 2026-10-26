#!/usr/bin/env python3
"""Derive computed design parameters from step1 config.json."""

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


def _repo_rel(repo: Path, path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    return p.resolve().relative_to(repo.resolve()).as_posix()


def _require_id(obj: dict[str, Any], key: str, ctx: str) -> int:
    val = obj.get(key)
    if type(val) is not int or val < 0:
        _die(f"{ctx}: missing {key} (non-negative int required)")
    return val


def _core_threads(core: dict[str, Any]) -> list[dict[str, Any]]:
    threads = core["threads"]
    if not isinstance(threads, list):
        _die("threads must be an explicit instance list")
    ids = [_require_id(thread, "id", "thread") for thread in threads]
    if len(set(ids)) != len(ids):
        _die("duplicate thread ID within core")
    return threads


def configured_cores(design: dict[str, Any]) -> list[dict[str, Any]]:
    cores = design["cores"]
    if not isinstance(cores, list):
        _die("design cores must be an explicit instance list")
    harts = [_require_id(core, "hartId", "core") for core in cores]
    if len(set(harts)) != len(harts):
        _die("duplicate hart ID")
    for core in cores:
        if "count" in core or "factory" in core:
            _die(
                "core instances must be declared directly; hardware is assembled by the design"
            )
    return cores


def iter_threads(topo: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for core in configured_cores(topo):
        yield from _core_threads(core)


def bank_params(thread: dict, pkg: str) -> tuple[int, int, int]:
    mem = thread.get("memdomain")
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


def _balldomain_base_dir(repo: Path, thread: dict, pkg: str) -> str:
    bd = thread.get("balldomain")
    if isinstance(bd, dict):
        rel = bd.get("_file")
        if isinstance(rel, str):
            return _repo_rel(repo, Path(rel).parent)
    raise ValueError(f"{pkg}: missing BallDomain configuration source")


def _ball_num(thread: dict) -> int:
    bd = normalize_ball_domain(thread)
    if not isinstance(bd, dict):
        return 0
    return len(bd["balls"])


def _derive_threads(repo: Path, topo: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, thread in enumerate(iter_threads(topo)):
        rel = thread.get("_file")
        if not isinstance(rel, str):
            _die("thread config missing _file")
        pkg = thread["component"]
        if not isinstance(pkg, str) or not pkg.isidentifier():
            _die(f"invalid component name: {pkg!r}")
        role = thread.get("name")
        if role is not None and not isinstance(role, str):
            _die("thread name must be a string")
        thread_id = _require_id(thread, "id", f"thread index {index}")
        ball_num = _ball_num(thread)
        if isinstance(thread.get("memdomain"), dict):
            num, width, entries = bank_params(thread, pkg)
        elif ball_num > 0:
            _die(f"{pkg}: buckyball thread missing memdomain")
        else:
            num, width, entries = 0, 0, 0
        entry: dict[str, Any] = {
            "index": index,
            "thread_id": thread_id,
            "pkg": pkg,
            "role": role or "",
            "profile": thread["compiler"]["name"],
            "cpu_kind": thread["cpu"]["kind"] if "cpu" in thread else "",
            "config_path": _repo_rel(repo, rel),
            "balldomain_base_dir": _balldomain_base_dir(repo, thread, pkg),
            "bank_num": num,
            "bank_width": width,
            "bank_entries": entries,
            "ball_num": ball_num,
        }
        if ball_num > 0:
            bd = normalize_ball_domain(thread)
            if not isinstance(bd, dict):
                _die(f"{pkg}: missing balldomain")
            entry["balls"] = [
                {"ball_id": ball["id"], "ball_dir": ball["component"]}
                for ball in bd["balls"]
            ]
        out.append(entry)
    return out


def _validate_spm(config: dict[str, Any]) -> None:
    base, size, bits = (config[key] for key in ("base", "bytes", "dataBits"))
    if any(type(value) is not int for value in (base, size, bits)):
        _die("SPM base/bytes/dataBits must be integers")
    if bits < 64 or bits & (bits - 1):
        _die("SPM dataBits must be a power of two >= 64")
    beat = bits // 8
    if (
        size < 2 * beat
        or size & (size - 1)
        or base < 0
        or base % beat
        or base + size > 1 << 64
    ):
        _die(
            "SPM requires an aligned base and power-of-two capacity within the 64-bit address space"
        )


def _derive_cores(
    repo: Path, topo: dict[str, Any], threads: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    placements: list[dict[str, Any]] = []
    offset = 0
    for core in configured_cores(topo):
        core_path = core.get("_file")
        if not isinstance(core_path, str):
            _die("core missing _file")
        core_threads = _core_threads(core)
        n = len(core_threads)
        ants = [
            thread
            for thread in core_threads
            if "cpu" in thread and thread["cpu"]["kind"] == "ant"
        ]
        if ants:
            shared_spm = core.get("tss")
            if not isinstance(shared_spm, dict):
                _die("Ant compute core requires explicit tss geometry")
            _validate_spm(shared_spm)
            for thread in ants:
                params = thread["cpu"]["config"]
                tls = params["tls"]
                _validate_spm(tls)
                code = params["codeBytes"]
                if type(code) is not int or code < 32 or code & (code - 1):
                    _die("Ant codeBytes must be a power of two >= 32")
                if not 1 <= params["taskBits"] <= 64:
                    _die("Ant taskBits must be in [1,64]")
                if min(tls["base"], shared_spm["base"]) < code:
                    _die("Ant code overlaps TLS/TSS")
                if not (
                    tls["base"] + tls["bytes"] <= shared_spm["base"]
                    or shared_spm["base"] + shared_spm["bytes"] <= tls["base"]
                ):
                    _die("Ant TLS/TSS windows overlap")
                if tls["dataBits"] != shared_spm["dataBits"]:
                    _die("Ant TLS/TSS access widths must match")
        shared = core.get("sharedMem")
        if isinstance(shared, dict):
            if shared["bankWidth"] != 128:
                _die("sharedMem.bankWidth must be 128 bits")
            rows = shared["bankEntries"]
            if type(rows) is not int or rows < 2 or rows > 65536 or rows & (rows - 1):
                _die("sharedMem.bankEntries must be a power of two in [2,65536]")
            if shared["enable"] and (
                shared["entries"] <= 0 or shared["entries"] % rows
            ):
                _die("sharedMem.entries must be a positive multiple of bankEntries")
        vbc = _require_id(shared, "virtualBankCount", "shared memory")
        indices = list(range(offset, offset + n))
        placements.append(
            {
                "hart_id": core["hartId"],
                "path": _repo_rel(repo, core_path),
                "thread_indices": indices,
                "threads_per_core": n,
                "virtual_bank_count": vbc,
            }
        )
        offset += n
    return placements


def _bemu_balls(repo: Path, profiles: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    balls: list[dict[str, str]] = []
    for profile in profiles:
        bd = normalize_ball_domain(profile["config"])
        if not isinstance(bd, dict):
            continue
        for ball in bd["balls"]:
            instructions = ball["instructions"]
            if any(entry["funct7"] == 0 for entry in instructions):
                _die("funct7 zero is not a Buckyball instruction")
            if not any(
                entry["funct7"] not in {1, 16, 32, 33, 34, 35} for entry in instructions
            ):
                continue
            ball_dir = ball["component"]
            ball_class = f"examples.balls.{ball_dir}.{ball['name']}"
            if ball_class in seen:
                continue
            seen.add(ball_class)
            emu_lib = (
                repo
                / "examples"
                / "components"
                / "balls"
                / ball_dir
                / "emu"
                / "src"
                / "lib.rs"
            )
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


def _ball_ctest_dirs(repo: Path, thread: dict[str, Any]) -> list[str]:
    bd = normalize_ball_domain(thread)
    if not isinstance(bd, dict):
        _die("thread missing balldomain for ctest dirs")
    dirs: list[str] = []
    for ball in bd["balls"]:
        path = (
            repo
            / "examples"
            / "components"
            / "balls"
            / ball["component"]
            / "workloads"
            / "ctests"
        )
        if not path.is_dir():
            _die(f"missing {path}")
        dirs.append(_repo_rel(repo, path))
    return dirs


def compiler_profiles(device: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = []
    for core in configured_cores(device):
        for thread in core["threads"]:
            declaration = thread["compiler"]
            profiles.append(
                {
                    "name": declaration["name"],
                    "component": declaration["component"],
                    "config": thread,
                }
            )
    for core in configured_cores(device):
        if "accelerator" in core:
            declaration = core["compiler"]
            profiles.append(
                {
                    "name": declaration["name"],
                    "component": declaration["component"],
                    "config": core["accelerator"],
                }
            )
    return profiles


def _derive_targets(
    repo: Path, profiles: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    targets = {}
    for profile in profiles:
        name, pkg, config = profile["name"], profile["component"], profile["config"]
        num, width, entries = bank_params(config, pkg)
        targets[name] = {
            "pkg": pkg,
            "compiler": "compiler",
            "bank": {"num": num, "width": width, "entries": entries},
            "ball_ctest_dirs": _ball_ctest_dirs(repo, config),
        }
    return targets


def _derive_harts(
    cores: list[dict[str, Any]], threads: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [{"hart_id": core["hart_id"]} for core in cores]


def _workload(
    design: str, targets: dict[str, dict[str, Any]], repo: Path
) -> dict[str, Any]:
    defs: dict[str, str] = {
        "BUCKYBALL_WORKLOAD_DESIGN": design,
    }
    return {"cmake_param": defs}


def derive(data: dict[str, Any], repo: Path, design: str) -> dict[str, Any]:
    repo = repo.resolve()
    topo = data.get("device")
    if not isinstance(topo, dict):
        _die(f"{design}: missing device")

    root_file = data.get("_file")
    if not isinstance(root_file, str):
        _die(f"{design}: missing root _file")
    design_file = topo.get("_file")
    if not isinstance(design_file, str):
        _die(f"{design}: device missing _file")

    includes_raw = data.get("includes")
    if not isinstance(includes_raw, list):
        _die(f"{design}: includes must be a list")

    threads = _derive_threads(repo, topo)
    cores = _derive_cores(repo, topo, threads)
    profiles = compiler_profiles(topo)
    targets = _derive_targets(repo, profiles)
    harts = _derive_harts(cores, threads)
    if sorted(hart["hart_id"] for hart in harts) != list(range(len(harts))):
        _die("complete design hart IDs must be contiguous from zero")
    n_cores = len(configured_cores(topo))

    return {
        "design": design,
        "name": Path(root_file).parent.parent.name,
        "design_path": _repo_rel(repo, root_file),
        "core_config_path": _repo_rel(repo, design_file),
        "includes": [_repo_rel(repo, p) for p in includes_raw],
        "n_cores": n_cores,
        "targets": targets,
        "harts": harts,
        "threads": threads,
        "cores": cores,
        "bemu": {
            "balls": _bemu_balls(repo, profiles),
        },
        "workload": _workload(design, targets, repo),
    }


def write_derived(
    data: dict[str, Any], repo: Path, design: str, out: Path
) -> dict[str, Any]:
    derived = derive(data, repo, design)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return derived


def main() -> None:
    if len(sys.argv) not in (4, 5):
        _die(f"usage: {sys.argv[0]} REPO DESIGN CONFIG.json [OUT.json]")
    repo = Path(sys.argv[1])
    design = sys.argv[2]
    config = Path(sys.argv[3])
    out = Path(sys.argv[4]) if len(sys.argv) == 5 else config.parent / "derived.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    derived = write_derived(data, repo, design, out)
    sys.stdout.write(json.dumps(derived, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
