import json
import os
import re
import subprocess
from pathlib import Path

_PRODUCT = {"verilog": "verilator", "synth": "verilator", "p2e": "p2e"}


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


def chip_arch_root(bbdir, chip):
    return os.path.join(bbdir, "arch", "build", _chip_name(chip))


def sim_name(bbdir, chip, product):
    if product not in _PRODUCT:
        raise ValueError(f"invalid rtl product: {product}")
    path = (
        Path(bbdir)
        / "examples"
        / "chips"
        / _chip_name(chip)
        / "configs"
        / "generated"
        / "config"
        / "config.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["sims"][_PRODUCT[product]]


def rtl_dir(bbdir, chip, product, output_dir=None):
    if output_dir:
        return output_dir
    return os.path.join(chip_arch_root(bbdir, chip), sim_name(bbdir, chip, product))


def rtl_out(bbdir, chip, product, output_dir=None):
    name = sim_name(bbdir, chip, product)
    out = output_dir or os.path.join(chip_arch_root(bbdir, chip), name)
    return name, out


def _log_part(value, what):
    if not isinstance(value, str) or not value:
        raise ValueError(f"log dir missing {what}")
    if os.path.basename(value) != value:
        raise ValueError(f"log {what} must be a single path component: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"invalid log {what}: {value!r}")
    return value


def log_dir(bbdir, chip, product, stamp, tool, name, output_dir=None):
    chip = _chip_name(chip)
    config = _log_part(
        os.path.basename(os.path.normpath(rtl_dir(bbdir, chip, product, output_dir))),
        "config",
    )
    stamp = _log_part(stamp, "timestamp")
    tool = _log_part(tool, "tool")
    name = _log_part(os.path.basename(name), "name")
    return os.path.join(bbdir, "log", f"{stamp}-{chip}-{config}-{tool}-{name}")
