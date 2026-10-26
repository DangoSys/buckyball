from __future__ import annotations

import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2] / "dc" / "scripts") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dc" / "scripts"))

from tapeout import SramGeom, get_tapeout_contract
from replace import source_scope
from sram_compiler import generate_sram_dbs, generate_sram_macros


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


def geoms_from_names(
    *,
    names: list[str],
    sram_table: dict[str, SramGeom],
    mdf: Path,
) -> list[SramGeom]:
    index = load_mdf(mdf)
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


def check_bitwrite_contract(
    geoms: list[SramGeom], mdf: Path, sram_table: dict[str, SramGeom]
) -> None:
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
                raise ValueError(
                    f"{g.name}: bitwrite requires mask port BWEN, got {mask_name!r}"
                )
            if p.get("mask port polarity") != "active low":
                raise ValueError(f"{g.name}: BWEN polarity must be active low")
            if p.get("mask granularity") != 1:
                raise ValueError(f"{g.name}: BWEN mask granularity must be 1")
        elif mask_name is not None:
            raise ValueError(f"{g.name}: mask port {mask_name!r} but bitwrite is false")
    used = {g.name for g in geoms}
    for geom in sram_table.values():
        if geom.bitwrite and geom.name not in used:
            raise RuntimeError(
                f"bitwrite sram {geom.name} not instantiated by any memory"
            )


def _leaf_paths(cache_dir: Path, g: SramGeom, corner: str) -> dict[str, str]:
    name = g.name
    leaf = cache_dir / f"{name}-mux{g.mux}"
    return {
        "v": str((leaf / f"{name}.v").resolve()),
        "lib": str((leaf / f"{name}_{corner}.lib").resolve()),
        "db": str((leaf / f"{name}_{corner}.db").resolve()),
    }


def generate_sram(
    *,
    bbdir: str | Path,
    design: str,
    selection: str,
    build_dir: str | Path,
    top_module: str,
    source_list: str | Path,
    out_dir: str | Path | None = None,
) -> dict:
    bbdir = Path(bbdir).resolve()
    build_dir = Path(build_dir).resolve()
    src = build_dir / "mems.conf"
    if not src.is_file():
        raise FileNotFoundError(f"missing elaborator mems.conf: {src}")
    mems = src.read_text()
    reachable, _ = source_scope(Path(source_list).read_text().splitlines(), top_module)
    selected = "".join(
        line + "\n"
        for line in mems.splitlines()
        if line.strip() and line.split()[1] in reachable
    )
    contract = get_tapeout_contract(bbdir, design, selection=selection)
    if out_dir is None:
        dest = build_dir / "ip-generate"
    else:
        if isinstance(out_dir, str) and not out_dir.strip():
            raise ValueError("empty out_dir")
        dest = Path(out_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    mems_out = dest / "mems.conf"
    verilog = dest / "sram_macros.v"
    mems_out.write_text(selected)
    if selected:
        text, names = generate_sram_macros(
            process=contract.sram_process, mems=selected, mdf=contract.sram_mdf
        )
        verilog.write_text(text)
        geoms = geoms_from_names(
            names=names,
            sram_table=contract.sram_table,
            mdf=contract.sram_mdf,
        )
        check_bitwrite_contract(geoms, contract.sram_mdf, contract.sram_table)
    else:
        verilog.write_text("")
        geoms = []
    ip_db = build_dir.parent / f"{build_dir.name}-ip-db"
    if geoms:
        db_paths, corner_tag = generate_sram_dbs(
            geoms=geoms,
            process=contract.sram_process,
            corner=contract.sram_corner,
            cache_dir=ip_db,
            lc_shell=contract.lc_shell,
        )
    else:
        db_paths, corner_tag = [], contract.sram_corner
    names = [g.name for g in geoms]
    man = {
        "design": design,
        "sram_mdf": str(contract.sram_mdf),
        "mems_conf": str(mems_out),
        "sram_macros_v": str(verilog),
        "sram_corner": corner_tag,
        "ip_db": str(ip_db),
        "leaves": names,
        "leaf_paths": {g.name: _leaf_paths(ip_db, g, corner_tag) for g in geoms},
        "link_dbs": [str(p) for p in db_paths],
        "leaf_count": len(names),
        "mem_count": len(selected.splitlines()),
    }
    manifest_path = dest / "generate_manifest.json"
    manifest_path.write_text(json.dumps(man, indent=2) + "\n")
    return {**man, "generate_manifest": str(manifest_path)}
