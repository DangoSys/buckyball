from __future__ import annotations

import sys
from pathlib import Path

for path in (
    Path(__file__).resolve().parents[2],
    Path(__file__).resolve().parents[5] / "thirdparty" / "soc-framework" / "ip",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from smic180.compiler import generate_smic180_sram_dbs
from smic180.sram import smic180_sram


def generate_sram_macros(
    *, process: str, mems: str, mdf: Path
) -> tuple[str, list[str]]:
    if process == "smic180":
        return smic180_sram(mems, mdf)
    raise ValueError(f"unknown sram process {process!r}")


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
