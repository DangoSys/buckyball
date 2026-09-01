from __future__ import annotations

import json
import sys
from pathlib import Path

_DC = Path(__file__).resolve().parents[2] / "dc" / "scripts"
if str(_DC) not in sys.path:
    sys.path.insert(0, str(_DC))

from tapeout import SramGeom, get_tapeout_contract
from macro_compiler import run_macro_compiler
from sram_compiler import generate_sram_dbs, leaf_names_from_macros


def pad_mems_conf(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        lines.append(raw.rstrip() + " ")
    if not lines:
        raise ValueError("empty mems.conf")
    return "\n".join(lines) + "\n"


def geoms_from_macros(
    *,
    macros_v: Path,
    sram_table: dict[str, SramGeom],
    mdf: Path,
) -> list[SramGeom]:
    raw = json.loads(mdf.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"mdf must be a list: {mdf}")
    index: dict[str, dict] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"mdf entry must be object: {mdf}")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"mdf entry missing name: {mdf}")
        if name in index:
            raise ValueError(f"duplicate mdf name {name}: {mdf}")
        index[name] = entry

    names = leaf_names_from_macros(macros_v, set(sram_table) | set(index))
    geoms: list[SramGeom] = []
    for name in names:
        if name not in sram_table:
            raise ValueError(f"sram leaf {name} missing from tapeout.sram")
        if name not in index:
            raise ValueError(f"sram leaf {name} missing from mdf: {mdf}")
        entry = index[name]
        try:
            depth = int(entry["depth"])
            width = int(entry["width"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"mdf entry {name} missing depth/width") from exc
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
    geoms = geoms_from_macros(
        macros_v=verilog,
        sram_table=contract.sram_table,
        mdf=contract.sram_mdf,
    )
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
