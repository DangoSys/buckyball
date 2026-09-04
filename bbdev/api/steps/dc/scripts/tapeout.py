"""Resolve chip-owned tapeout flow contracts."""

from __future__ import annotations

import os
import re
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path


def _tcl_word(value: str) -> str:
    return "{" + str(value).replace("\\", "\\\\").replace("}", "\\}") + "}"


def _tcl_list(values: list[str]) -> str:
    return "[list " + " ".join(_tcl_word(value) for value in values) + "]"


@dataclass(frozen=True)
class SramGeom:
    name: str
    words: int
    mux: int
    bits: int
    bitwrite: bool = False


@dataclass(frozen=True)
class TapeoutContract:
    chip: str
    root: Path
    dc_script: Path
    constraints_sdc: Path
    power_script: Path
    power_sim_script: Path
    sram_mdf: Path
    top_module: str
    target_library: str
    synthetic_library: list[str]
    link_library: list[str]
    max_cores: int
    power_format: str
    power_start_ns: str | None
    power_end_ns: str | None
    power_workload: str | None
    power_strip_path: str
    sram_process: str
    sram_corner: str
    lc_shell: Path
    sram_table: dict[str, SramGeom]


def _p(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def resolve_power_window(
    contract: TapeoutContract, start_ns: object | None, end_ns: object | None
) -> tuple[str | None, str | None]:
    start = str(start_ns) if start_ns is not None and str(start_ns) != "" else contract.power_start_ns
    end = str(end_ns) if end_ns is not None and str(end_ns) != "" else contract.power_end_ns
    if (start is None) != (end is None):
        raise ValueError("power start_ns and end_ns must be supplied together")
    if start is not None:
        try:
            if float(start) < 0 or float(start) >= float(end):
                raise ValueError
        except ValueError as exc:
            raise ValueError("power start_ns must be smaller than end_ns") from exc
    return start, end


def clock_period_ns(contract: TapeoutContract) -> float:
    text = contract.constraints_sdc.read_text(encoding="utf-8")
    match = re.search(r"create_clock\b[^;\n]*-period\s+([0-9]+(?:\.[0-9]+)?)", text)
    if match is None:
        raise ValueError(f"no create_clock -period in {contract.constraints_sdc}")
    period = float(match.group(1))
    if period <= 0:
        raise ValueError(f"invalid create_clock -period in {contract.constraints_sdc}")
    return period


def get_tapeout_contract(bbdir: str | os.PathLike[str], chip: str, top: str | None = None) -> TapeoutContract:
    root = Path(bbdir) / "examples" / "chips" / chip / "tapeout"
    with (root / "config.toml").open("rb") as f:
        s = tomllib.load(f)["tapeout"]

    sram_table = {}
    for row in s["sram"]:
        name = row["name"]
        if name in sram_table:
            raise ValueError(f"duplicate tapeout.sram name {name}")
        bitwrite = row["bitwrite"] if "bitwrite" in row else False
        if not isinstance(bitwrite, bool):
            raise ValueError(f"tapeout.sram {name}: bitwrite must be bool, got {bitwrite!r}")
        sram_table[name] = SramGeom(name, row["words"], row["mux"], row["bits"], bitwrite)

    pw = s.get("power_workload")
    return TapeoutContract(
        chip=chip,
        root=root,
        dc_script=_p(root, s["dc_script"]),
        constraints_sdc=_p(root, s["constraints_sdc"]),
        power_script=_p(root, s["power_script"]),
        power_sim_script=_p(root, s["power_sim_script"]),
        sram_mdf=_p(root, s["sram_mdf"]),
        top_module=top or s["top"],
        target_library=str(_p(root, s["target_library"])),
        synthetic_library=[str(_p(root, p)) for p in s["synthetic_library"]],
        link_library=[str(_p(root, p)) for p in s.get("link_library", [])],
        max_cores=s["max_cores"],
        power_format=s["power_format"],
        power_start_ns=s.get("power_start_ns"),
        power_end_ns=s.get("power_end_ns"),
        power_workload=None if pw in (None, "") else str(pw),
        power_strip_path=s.get("power_strip_path", ""),
        sram_process=s["sram_process"],
        sram_corner=s["sram_corner"],
        lc_shell=_p(root, s["lc_shell"]),
        sram_table=sram_table,
    )


def write_run_tcl(path: str | os.PathLike[str], values: dict[str, object]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in values.items():
        name = "RUN_" + key.upper()
        if isinstance(value, list):
            lines.append(f"set {name} {_tcl_list([str(item) for item in value])}")
        elif value is None:
            lines.append(f"set {name} {{}}")
        elif isinstance(value, (int, float)):
            lines.append(f"set {name} {value}")
        else:
            lines.append(f"set {name} {_tcl_word(str(value))}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_run_env(path: str | os.PathLike[str], values: dict[str, object]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={shlex.quote(str(value))}" for key, value in values.items() if value is not None]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
