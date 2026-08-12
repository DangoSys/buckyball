"""Resolve chip-owned tapeout flow contracts."""

from __future__ import annotations

import os
import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path

from utils.path import get_chip_from_config


def _tcl_word(value: str) -> str:
    return "{" + str(value).replace("\\", "\\\\").replace("}", "\\}") + "}"


def _tcl_list(values: list[str]) -> str:
    return "[list " + " ".join(_tcl_word(value) for value in values) + "]"


@dataclass(frozen=True)
class TapeoutContract:
    chip: str
    root: Path
    dc_script: Path
    power_script: Path
    power_sim_script: Path
    top_module: str
    clock_port: str
    clock_period_ns: float
    power_format: str
    power_start_ns: str | None
    power_end_ns: str | None
    power_workload: str | None
    power_strip_path: str


def resolve_power_window(
    contract: TapeoutContract, start_ns: object | None, end_ns: object | None
) -> tuple[str | None, str | None]:
    """Merge optional CLI window bounds with a chip default and validate them."""
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


def _read_config(root: Path) -> dict:
    path = root / "config.toml"
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    section = data.get("tapeout", data)
    if not isinstance(section, dict):
        raise ValueError(f"{path} must contain a [tapeout] table")
    return section


def get_tapeout_contract(bbdir: str | os.PathLike[str], config: str, top: str | None = None) -> TapeoutContract:
    """Resolve the tapeout scripts owned by the chip selected by a Scala config."""
    chip = get_chip_from_config(str(bbdir), config)
    root = Path(bbdir) / "examples" / "chips" / chip / "tapeout"
    if not root.is_dir():
        raise ValueError(f"chip {chip} has no tapeout directory: {root}")

    settings = _read_config(root)
    top_module = top or settings.get("top", "DigitalTop")
    clock_port = str(settings.get("clock_port", "")).strip()
    if not clock_port:
        raise ValueError(f"missing tapeout.clock_port in {root / 'config.toml'}")
    try:
        clock_period_ns = float(settings.get("clock_period_ns"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid tapeout.clock_period_ns in {root / 'config.toml'}") from exc
    if clock_period_ns <= 0:
        raise ValueError(f"tapeout.clock_period_ns must be positive in {root / 'config.toml'}")

    power_format = str(settings.get("power_format", "fsdb")).lower()
    if power_format not in {"saif", "vcd", "fsdb"}:
        raise ValueError(f"unsupported tapeout.power_format in {root / 'config.toml'}: {power_format}")

    def optional_string(name: str) -> str | None:
        value = settings.get(name)
        return None if value is None or str(value).strip() == "" else str(value)

    power_start_ns = optional_string("power_start_ns")
    power_end_ns = optional_string("power_end_ns")
    if (power_start_ns is None) != (power_end_ns is None):
        raise ValueError(f"tapeout power_start_ns and power_end_ns must be set together in {root / 'config.toml'}")
    if power_start_ns is not None:
        try:
            if float(power_start_ns) < 0 or float(power_start_ns) >= float(power_end_ns):
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"invalid tapeout power window in {root / 'config.toml'}") from exc

    scripts = {
        "dc_script": root / str(settings.get("dc_script", "dc.tcl")),
        "power_script": root / str(settings.get("power_script", "power.tcl")),
        "power_sim_script": root / str(settings.get("power_sim_script", "power_sim.sh")),
    }
    missing = [str(path) for path in scripts.values() if not path.is_file()]
    if missing:
        raise ValueError(f"chip {chip} tapeout contract is missing: {', '.join(missing)}")

    return TapeoutContract(
        chip=chip,
        root=root,
        dc_script=scripts["dc_script"],
        power_script=scripts["power_script"],
        power_sim_script=scripts["power_sim_script"],
        top_module=str(top_module),
        clock_port=clock_port,
        clock_period_ns=clock_period_ns,
        power_format=power_format,
        power_start_ns=power_start_ns,
        power_end_ns=power_end_ns,
        power_workload=optional_string("power_workload"),
        power_strip_path=str(settings.get("power_strip_path", "")),
    )


def technology_settings() -> dict[str, object]:
    """Read the small technology contract exported by the host setup."""
    target = os.environ.get("TARGET_LIBRARY", "").strip()
    synthetic = [item for item in os.environ.get("SYNTHETIC_LIBRARY", "").split(os.pathsep) if item]
    link = [item for item in os.environ.get("LINK_LIBRARY", "").split(os.pathsep) if item]
    if not target:
        raise ValueError("missing TARGET_LIBRARY; source sourceme_host.sh")
    if not Path(target).is_file():
        raise ValueError(f"TARGET_LIBRARY does not exist: {target}")
    missing = [item for item in synthetic + link if not Path(item).is_file()]
    if missing:
        raise ValueError("technology library does not exist: " + ", ".join(missing))
    if not synthetic:
        raise ValueError("missing SYNTHETIC_LIBRARY; source sourceme_host.sh")
    return {
        "target_library": target,
        "synthetic_library": synthetic,
        "link_library": link,
        "max_cores": 8,
    }


def write_run_tcl(path: str | os.PathLike[str], values: dict[str, object]) -> Path:
    """Write a Tcl variable file consumed by chip-owned DC/PT scripts."""
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
    """Write shell assignments for a chip-owned power simulation wrapper."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={shlex.quote(str(value))}" for key, value in values.items() if value is not None]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
