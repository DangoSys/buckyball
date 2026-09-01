from __future__ import annotations

import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_API = _HERE.parents[2]
_IP = _HERE.parents[4] / "thirdparty" / "soc-framework" / "ip"
for _p in (str(_API), str(_IP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from smic180.compiler import generate_smic180_sram_dbs


def leaf_names_from_macros(macros_v: Path, known: set[str]) -> list[str]:
    if not macros_v.is_file():
        raise FileNotFoundError(f"missing macros verilog: {macros_v}")
    text = macros_v.read_text()
    ordered: list[str] = []
    for name in sorted(known):
        pat = re.compile(rf"(?m)^\s*{re.escape(name)}\s+[A-Za-z_][A-Za-z0-9_$]*\s*\(")
        if pat.search(text):
            ordered.append(name)
    if not ordered:
        raise RuntimeError(f"no known sram leaf instances in {macros_v}")
    return ordered


def generate_sram_dbs(
    *,
    geoms: list,
    process: str,
    corner: str,
    cache_dir: Path,
    lc_shell: Path,
) -> tuple[list[Path], str]:
    if process == "smic180":
        return generate_smic180_sram_dbs(geoms, corner, cache_dir, lc_shell)
    elif process == "tsmc28":
        raise ValueError("tsmc28 sram is not supported yet")
    raise ValueError(f"unknown sram process {process!r}")
