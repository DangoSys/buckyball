import json
import os
import re
import sys
from pathlib import Path


def get_buckyball_path():
    current_dir = os.path.dirname(__file__)
    root = os.path.realpath(
        os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    )
    home = os.path.realpath(str(Path.home()))
    if not Path(root).is_relative_to(home):
        raise ValueError(f"buckyball root must be under $HOME ({home}): {root}")
    return root


def _design_name(design):
    if not isinstance(design, str) or not design or design == "None":
        raise ValueError("missing required parameter: design")
    if not re.fullmatch("[A-Za-z0-9_.-]+", design):
        raise ValueError(f"invalid design name: {design}")
    return design


def design_output_root(bbdir, design, *, selection):
    design = _design_name(design)
    return os.path.join(bbdir, "examples/designs", design, "output/sims", selection)


def workloads_output_root(bbdir):
    return os.path.join(bbdir, "bb-tests", "output")


def workload_build_dir(bbdir, design, *, selection):
    design = _design_name(design)
    return os.path.join(
        design_output_root(bbdir, design, selection=selection), "workloads-build"
    )


def workload_tests_root(bbdir, design, *, selection):
    return os.path.join(
        design_output_root(bbdir, design, selection=selection), "workloads", "src"
    )


def bebop_target_dir(bbdir, design, backend, *, selection):
    design = _design_name(design)
    if backend not in {"bemu", "verilator", "p2e", "vcs"}:
        raise ValueError(f"unknown Bebop backend: {backend}")
    return os.path.join(
        bbdir,
        "examples",
        "designs",
        design,
        "output",
        "sims",
        selection,
        "emu",
        backend,
        "target",
    )


def bebop_bin_path(bbdir, design, backend, *, selection):
    design = _design_name(design)
    if backend not in {"bemu", "verilator", "p2e", "vcs"}:
        raise ValueError(f"unknown Bebop backend: {backend}")
    return os.path.join(
        bbdir,
        "examples",
        "designs",
        design,
        "output",
        "sims",
        selection,
        "bin",
        f"bebop-{backend}",
    )


def design_arch_root(bbdir, design, *, selection):
    return os.path.join(
        bbdir,
        "examples/designs",
        _design_name(design),
        "output/sims",
        selection,
        "arch",
    )


def load_design_config(bbdir, design):
    sys.path.insert(0, str(Path(bbdir) / "bbdev/api/steps/config/scripts"))
    import design_pb2

    path = (
        Path(bbdir)
        / "examples/designs"
        / _design_name(design)
        / "output/generated/design.pb"
    )
    return design_pb2.Design.FromString(path.read_bytes())


def simulation_target(bbdir, design, selection):
    config = load_design_config(bbdir, design)
    if selection not in config.simulations:
        raise ValueError(
            f"design {design!r} has no registered simulation entry {selection!r}"
        )
    return config.simulations[selection]


def runner_manifest(bbdir, design, selection):
    """The simulation entry's emu crate: its model and the runners built from it."""
    return Path(bbdir) / simulation_target(bbdir, design, selection).emu / "Cargo.toml"


def design_resource_dir(bbdir, design, name, *, selection):
    resources = simulation_target(bbdir, design, selection).resources
    if name not in resources:
        raise ValueError(f"{design}/{selection} has no registered {name} directory")
    return Path(bbdir) / resources[name]


def sim_name(bbdir, design, product, *, selection):
    target = simulation_target(bbdir, design, selection)
    return "P2ETop" if product == "p2e" else target.top


def rtl_dir(bbdir, design, product, output_dir=None, *, selection):
    # Each product (verilog, tapeout, dc, p2e) elaborates its own RTL.
    name = sim_name(bbdir, design, product, selection=selection)
    return output_dir or os.path.join(
        design_arch_root(bbdir, design, selection=selection), product, name
    )


def rtl_out(bbdir, design, product, output_dir=None, *, selection):
    name = sim_name(bbdir, design, product, selection=selection)
    return name, rtl_dir(bbdir, design, product, output_dir, selection=selection)


def _log_part(value, what):
    if not isinstance(value, str) or not value:
        raise ValueError(f"log dir missing {what}")
    if os.path.basename(value) != value:
        raise ValueError(f"log {what} must be a single path component: {value!r}")
    if not re.fullmatch("[A-Za-z0-9_.-]+", value):
        raise ValueError(f"invalid log {what}: {value!r}")
    return value


def log_dir(
    bbdir, design, product, stamp, tool, name, output_dir=None, *, selection=None
):
    design = _design_name(design)
    stamp = _log_part(stamp, "timestamp")
    tool = _log_part(tool, "tool")
    name = _log_part(os.path.basename(name), "name")
    if selection is not None:
        name = _log_part(selection.replace("/", "-"), "target") + "-" + name
    return os.path.join(bbdir, "log", f"{stamp}-{design}-{tool}-{name}")


def run_log_dir(bbdir, run):
    root = (Path(bbdir) / "log").resolve()
    path = (root / _log_part(run, "run")).resolve()
    if path.parent != root:
        raise ValueError(f"run must name a directory under {root}: {run!r}")
    return path
