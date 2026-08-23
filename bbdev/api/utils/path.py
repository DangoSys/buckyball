import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from utils.chip import chip_field, chip_toml


def get_buckyball_path():
    current_dir = os.path.dirname(__file__)
    # bbdev/api/utils -> bbdev/api -> bbdev -> buckyball
    inferred = os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))

    root = os.environ.get("BUCKYBALL_ROOT")
    if root:
        if not os.path.isabs(root):
            raise ValueError("BUCKYBALL_ROOT must be an absolute path")
        if not os.path.isdir(root):
            raise ValueError(f"BUCKYBALL_ROOT does not exist: {root}")
        root = os.path.realpath(root)
        if root != inferred:
            raise ValueError(
                f"BUCKYBALL_ROOT={root} does not match bbdev tree root {inferred}"
            )
    else:
        root = inferred

    home = os.path.realpath(str(Path.home()))
    if not Path(root).is_relative_to(home):
        raise ValueError(f"buckyball root must be under $HOME ({home}): {root}")
    return root


def _chip_name(chip):
    if not isinstance(chip, str) or not chip or chip == "None":
        raise ValueError("missing required parameter: chip")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", chip):
        raise ValueError(f"invalid chip name: {chip}")
    return chip


def chip_output_root(bbdir, chip):
    chip = _chip_name(chip)
    return os.path.join(bbdir, "bb-tests", "output", chip)


def workload_build_dir(bbdir, chip):
    chip = _chip_name(chip)
    return os.path.join(bbdir, "bb-tests", "workloads", "build", chip)


def workload_tests_root(bbdir, chip):
    return os.path.join(chip_output_root(bbdir, chip), "workloads", "src")


def bebop_target_dir(bbdir, chip):
    chip = _chip_name(chip)
    return os.path.join(bbdir, "bebop", "target", chip)


def bebop_cargo_env(bbdir, chip):
    return {"CARGO_TARGET_DIR": bebop_target_dir(bbdir, chip)}


def gcc_lib_dir(soname: str) -> str:
    candidates = []
    printed = subprocess.run(
        ["g++", f"-print-file-name={soname}"],
        capture_output=True,
        text=True,
    )
    printed_path = (printed.stdout or "").strip()
    if printed.returncode == 0 and printed_path and "/" in printed_path:
        candidates.append(printed_path)
    for directory in os.environ.get("LIBRARY_PATH", "").split(":"):
        if directory:
            candidates.append(os.path.join(directory, soname))
    for match in re.finditer(r"-L(\S+)", os.environ.get("NIX_LDFLAGS", "")):
        candidates.append(os.path.join(match.group(1), soname))

    runtime_names = [soname]
    if soname.endswith(".so"):
        runtime_names.append(soname + ".1")

    for path in candidates:
        if not os.path.isfile(path):
            continue
        lib_dir = os.path.dirname(os.path.realpath(path))
        if any(os.path.isfile(os.path.join(lib_dir, name)) for name in runtime_names):
            return lib_dir

    detail = printed_path or (printed.stderr or "").strip() or f"exit {printed.returncode}"
    raise RuntimeError(
        f"cannot locate {soname} (g++ -print-file-name={soname} -> {detail!r}; "
        f"LIBRARY_PATH={os.environ.get('LIBRARY_PATH', '')!r}; "
        f"NIX_LDFLAGS={os.environ.get('NIX_LDFLAGS', '')!r})"
    )


_RTL_FIELD = {
    "verilog": "verilatorConfig",
    "synth": "verilatorConfig",
    "p2e": "p2eConfig",
}


def chip_arch_root(bbdir, chip):
    return os.path.join(bbdir, "arch", "build", _chip_name(chip))


def rtl_dir(bbdir, chip, product, output_dir=None):
    if output_dir:
        return output_dir
    if product not in _RTL_FIELD:
        raise ValueError(f"invalid rtl product: {product}")
    cfg = chip_field(bbdir, chip, _RTL_FIELD[product])
    return os.path.join(chip_arch_root(bbdir, chip), cfg)


def rtl_dir_for_clean(bbdir, chip, product, output_dir=None):
    return rtl_dir(bbdir, chip, product, output_dir)


def rtl_config_name(bbdir, chip, product, output_dir=None):
    return os.path.basename(os.path.normpath(rtl_dir(bbdir, chip, product, output_dir)))


def _log_part(value, what):
    if not isinstance(value, str) or not value:
        raise ValueError(f"log dir missing {what}")
    if os.path.basename(value) != value:
        raise ValueError(f"log {what} must be a single path component: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"invalid log {what}: {value!r}")
    return value


def get_run_log_dir(bbdir, chip, product, stamp, tool, name, output_dir=None):
    chip = _chip_name(chip)
    config = _log_part(rtl_config_name(bbdir, chip, product, output_dir), "config")
    stamp = _log_part(stamp, "timestamp")
    tool = _log_part(tool, "tool")
    name = _log_part(os.path.basename(name), "name")
    return os.path.join(bbdir, "log", f"{stamp}-{chip}-{config}-{tool}-{name}")


def get_bemu_log_dir(bbdir, chip, stamp, name):
    chip = _chip_name(chip)
    data = chip_toml(bbdir, chip)
    table = data.get("chip")
    if not isinstance(table, dict):
        raise ValueError(f"{chip}: missing [chip] in chip.toml")
    verilator = table.get("verilatorConfig")
    p2e = table.get("p2eConfig")
    if isinstance(verilator, str) and verilator:
        config = verilator
    elif isinstance(p2e, str) and p2e:
        config = p2e
    else:
        raise ValueError(f"{chip}: missing [chip].verilatorConfig and [chip].p2eConfig")
    stamp = _log_part(stamp, "timestamp")
    name = _log_part(os.path.basename(name), "name")
    config = _log_part(config, "config")
    return os.path.join(bbdir, "log", f"{stamp}-{chip}-{config}-bemu-{name}")


def get_verilator_build_dir(bbdir, chip=None, output_dir=None):
    if output_dir:
        return output_dir
    if chip is None:
        raise ValueError("missing required parameter: chip")
    return rtl_dir(bbdir, chip, "verilog")


def get_p2e_build_dir(bbdir, chip, output_dir=None):
    return rtl_dir(bbdir, chip, "p2e", output_dir)


def get_vcs_build_dir(bbdir, chip=None, output_dir=None):
    return get_verilator_build_dir(bbdir, chip, output_dir)


def get_arch_build_dir(bbdir, chip, output_dir=None):
    return rtl_dir(bbdir, chip, "synth", output_dir)


def get_dc_rtl_dir(bbdir, chip=None):
    return get_arch_build_dir(bbdir, chip)


def get_dc_analysis_dir(bbdir, chip=None, stage="area"):
    if stage not in {"area", "power"}:
        raise ValueError(f"invalid DC analysis stage: {stage}")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return get_run_log_dir(bbdir, chip, "synth", timestamp, "dc", stage)


def check_dc_rtl_args(body: dict):
    allowed = {"chip", "top"}
    for name in body:
        if name not in allowed:
            raise ValueError(f"unexpected parameter: {name}")
    top = body.get("top")
    if top is not None and (not isinstance(top, str) or not top):
        raise ValueError("top must be a non-empty module name")
    if isinstance(top, str) and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", top):
        raise ValueError("top must be a legal unescaped Verilog module name")


def check_dc_power_args(body: dict):
    allowed = {"chip", "top", "activity", "format", "strip_path", "workload", "start_ns", "end_ns", "start-ns", "end-ns"}
    for name in body:
        if name not in allowed:
            raise ValueError(f"unexpected parameter: {name}")
    common = {key: body[key] for key in ("chip", "top") if key in body}
    check_dc_rtl_args(common)
    activity = body.get("activity")
    activity_format = body.get("format")
    if activity is not None and (not isinstance(activity, str) or not activity):
        raise ValueError("activity must be a non-empty path when provided")
    if activity_format is not None and activity_format not in {"saif", "vcd", "fsdb"}:
        raise ValueError("format must be one of saif, vcd, or fsdb")
    if (activity is None) != (activity_format is None):
        raise ValueError("activity and format must be provided together")
    strip_path = body.get("strip_path")
    if strip_path is not None and not isinstance(strip_path, str):
        raise ValueError("strip_path must be a string")
    for name in ("workload",):
        value = body.get(name)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{name} must be a non-empty path when provided")
        if isinstance(value, str) and os.path.isabs(value):
            raise ValueError(f"{name} must be a built ELF name, not an absolute path")
    for name in ("start_ns", "end_ns", "start-ns", "end-ns"):
        value = body.get(name)
        if value is not None:
            try:
                if float(value) < 0:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a non-negative number") from exc
    start = body.get("start_ns", body.get("start-ns"))
    end = body.get("end_ns", body.get("end-ns"))
    if start is not None and end is not None and float(start) >= float(end):
        raise ValueError("start_ns must be smaller than end_ns")
