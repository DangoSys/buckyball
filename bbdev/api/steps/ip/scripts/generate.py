from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

_DC = Path(__file__).resolve().parents[2] / "dc" / "scripts"
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

from tapeout import SramGeom, get_tapeout_contract
from macro_compiler import run_macro_compiler
from sram_compiler import generate_sram_dbs, leaf_names_from_macros

_MODULE = re.compile(r"(?m)^module\s+(\w+)\s*\(")
_INST = re.compile(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_$]*)\s+[A-Za-z_][A-Za-z0-9_$]*\s*\(")


def pad_mems_conf(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        lines.append(raw.rstrip() + " ")
    if not lines:
        raise ValueError("empty mems.conf")
    return "\n".join(lines) + "\n"


def load_mdf(mdf: Path) -> dict[str, tuple[int, int]]:
    raw = json.loads(mdf.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"mdf must be a list: {mdf}")
    index: dict[str, tuple[int, int]] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"mdf entry must be object: {mdf}")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"mdf entry missing name: {mdf}")
        if name in index:
            raise ValueError(f"duplicate mdf name {name}: {mdf}")
        try:
            depth = int(entry["depth"])
            width = int(entry["width"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"mdf entry {name} missing depth/width") from exc
        if depth <= 0 or width <= 0:
            raise ValueError(f"mdf entry {name}: depth/width must be positive")
        index[name] = (depth, width)
    return index


def parse_mems(text: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6 or parts[0] != "name" or parts[2] != "depth" or parts[4] != "width":
            raise ValueError(f"mems.conf:{i}: expected 'name <id> depth <n> width <n> ...', got {line!r}")
        try:
            depth = int(parts[3])
            width = int(parts[5])
        except ValueError as exc:
            raise ValueError(f"mems.conf:{i}: depth/width not int") from exc
        if depth <= 0 or width <= 0:
            raise ValueError(f"mems.conf:{i}: depth/width must be positive")
        out.append((parts[1], depth, width))
    if not out:
        raise ValueError("empty mems.conf")
    return out


def module_insts(text: str) -> dict[str, Counter[str]]:
    hits = list(_MODULE.finditer(text))
    out: dict[str, Counter[str]] = {}
    for i, m in enumerate(hits):
        name = m.group(1)
        if name in out:
            raise ValueError(f"duplicate module {name}")
        start = m.end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        counts: Counter[str] = Counter()
        for inst in _INST.finditer(text[start:end]):
            leaf = inst.group(1)
            if leaf in ("module", "assign", "wire", "input", "output"):
                raise ValueError(f"module {name}: refused instance parse {leaf!r}")
            counts[leaf] += 1
        out[name] = counts
    return out


def check_mapping(*, mems_text: str, macros_v: Path, mdf: dict[str, tuple[int, int]]) -> None:
    if not macros_v.is_file():
        raise FileNotFoundError(f"missing macros verilog: {macros_v}")
    insts = module_insts(macros_v.read_text())
    lib = ", ".join(f"{n}={d}x{w}" for n, (d, w) in sorted(mdf.items()))
    errs: list[str] = []
    for name, depth, width in parse_mems(mems_text):
        need = depth * width
        if name not in insts:
            errs.append(f"{name}: missing module for {depth}x{width} ({need} bits)")
            continue
        counts = insts[name]
        if not counts:
            errs.append(f"{name}: empty mapping for {depth}x{width} ({need} bits)")
            continue
        got = 0
        parts = []
        unknown = False
        for leaf, n in sorted(counts.items()):
            if leaf not in mdf:
                errs.append(f"{name}: unknown leaf {leaf}")
                unknown = True
                break
            d, w = mdf[leaf]
            got += n * d * w
            parts.append(f"{n}x {leaf} ({d}x{w})")
        if unknown:
            continue
        if got != need:
            errs.append(
                f"{name}: need {depth}x{width} = {need} bits, "
                f"got {' + '.join(parts)} = {got} bits"
            )
    if errs:
        raise RuntimeError(
            "SRAM mapping mismatch:\n  " + "\n  ".join(errs) + f"\nlibrary: {lib}"
        )


def geoms_from_macros(
    *,
    macros_v: Path,
    sram_table: dict[str, SramGeom],
    mdf: Path,
) -> list[SramGeom]:
    index = load_mdf(mdf)
    names = leaf_names_from_macros(macros_v, set(sram_table) | set(index))
    geoms: list[SramGeom] = []
    for name in names:
        if name not in sram_table:
            raise ValueError(f"sram leaf {name} missing from tapeout.sram")
        if name not in index:
            raise ValueError(f"sram leaf {name} missing from mdf: {mdf}")
        depth, width = index[name]
        geom = sram_table[name]
        if depth != geom.words:
            raise ValueError(
                f"sram leaf {name}: mdf depth {depth} != tapeout.sram words {geom.words}"
            )
        if width != geom.bits:
            raise ValueError(
                f"sram leaf {name}: mdf width {width} != tapeout.sram bits {geom.bits}"
            )
        geoms.append(geom)
    return geoms


def check_bitwrite_contract(geoms: list[SramGeom], mdf: Path, macros_v: Path, sram_table: dict[str, SramGeom]) -> None:
    raw = json.loads(mdf.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"mdf must be a list: {mdf}")
    by_name: dict[str, dict] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"mdf entry must be object: {mdf}")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"mdf entry missing name: {mdf}")
        if name in by_name:
            raise ValueError(f"duplicate mdf name {name}: {mdf}")
        by_name[name] = entry
    for g in geoms:
        if g.name not in by_name:
            raise ValueError(f"sram leaf {g.name} missing from mdf: {mdf}")
        ports = by_name[g.name].get("ports")
        if not isinstance(ports, list) or len(ports) != 1:
            raise ValueError(f"{g.name}: expected exactly 1 mdf port")
        p = ports[0]
        if not isinstance(p, dict):
            raise ValueError(f"{g.name}: mdf port must be object")
        mask_name = p.get("mask port name")
        if g.bitwrite:
            if mask_name != "BWEN":
                raise ValueError(f"{g.name}: bitwrite requires mask port BWEN, got {mask_name!r}")
            if p.get("mask port polarity") != "active low":
                raise ValueError(f"{g.name}: BWEN polarity must be active low")
            if p.get("mask granularity") != 1:
                raise ValueError(f"{g.name}: BWEN mask granularity must be 1")
        elif mask_name is not None:
            raise ValueError(f"{g.name}: mask port {mask_name!r} but bitwrite is false")
    used = {g.name for g in geoms}
    for geom in sram_table.values():
        if geom.bitwrite and geom.name not in used:
            raise RuntimeError(f"bitwrite sram {geom.name} not instantiated in {macros_v}")


def _leaf_paths(cache_dir: Path, name: str, corner: str) -> dict[str, str]:
    leaf = cache_dir / name
    return {
        "v": str((leaf / f"{name}.v").resolve()),
        "lib": str((leaf / f"{name}_{corner}.lib").resolve()),
        "db": str((leaf / f"{name}_{corner}.db").resolve()),
    }


def generate_sram(
    *,
    bbdir: str | Path,
    chip: str,
    build_dir: str | Path,
    out_dir: str | Path | None = None,
) -> dict:
    bbdir = Path(bbdir).resolve()
    build_dir = Path(build_dir).resolve()
    src = build_dir / "mems.conf"
    if not src.is_file():
        raise FileNotFoundError(f"missing elaborator mems.conf: {src}")
    padded = pad_mems_conf(src.read_text())
    contract = get_tapeout_contract(bbdir, chip)
    if out_dir is None:
        dest = build_dir / "ip-generate"
    else:
        if isinstance(out_dir, str) and not out_dir.strip():
            raise ValueError("empty out_dir")
        dest = Path(out_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    mems_out = dest / "mems.conf"
    verilog = dest / "sram_macros.v"
    firrtl = dest / "sram_macros.fir"
    mems_out.write_text(padded)
    arch_dir = bbdir / "arch"
    run_macro_compiler(
        mems_conf=mems_out,
        mdf=contract.sram_mdf,
        verilog=verilog,
        firrtl=firrtl,
        arch_dir=arch_dir,
    )
    check_mapping(
        mems_text=padded,
        macros_v=verilog,
        mdf=load_mdf(contract.sram_mdf),
    )
    geoms = geoms_from_macros(
        macros_v=verilog,
        sram_table=contract.sram_table,
        mdf=contract.sram_mdf,
    )
    check_bitwrite_contract(geoms, contract.sram_mdf, verilog, contract.sram_table)
    ip_db = build_dir.parent / f"{build_dir.name}-ip-db"
    db_paths, corner_tag = generate_sram_dbs(
        geoms=geoms,
        process=contract.sram_process,
        corner=contract.sram_corner,
        cache_dir=ip_db,
        lc_shell=contract.lc_shell,
    )
    names = [g.name for g in geoms]
    man = {
        "chip": chip,
        "sram_mdf": str(contract.sram_mdf),
        "mems_conf": str(mems_out),
        "sram_macros_v": str(verilog),
        "sram_macros_fir": str(firrtl),
        "sram_corner": corner_tag,
        "ip_db": str(ip_db),
        "leaves": names,
        "leaf_paths": {
            g.name: _leaf_paths(ip_db, g.name, corner_tag)
            for g in geoms
        },
        "link_dbs": [str(p) for p in db_paths],
        "leaf_count": len(names),
        "mem_count": sum(1 for line in padded.splitlines() if line.strip()),
    }
    manifest_path = dest / "generate_manifest.json"
    manifest_path.write_text(json.dumps(man, indent=2) + "\n")
    return {**man, "generate_manifest": str(manifest_path)}
