import json
import os
import re
import sys
import time
from datetime import datetime
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
    """The owner of a simulation: a design name (examples/designs/<design>),
    or the path of a component alone relative to examples (prefab/balls/<ball>)."""
    if not isinstance(design, str) or not design or design == "None":
        raise ValueError("missing required parameter: design")
    if not re.fullmatch("[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*", design):
        raise ValueError(f"invalid design name: {design}")
    return design


def owner_root(bbdir, design):
    """The directory whose output/ holds the simulation's build products."""
    design = _design_name(design)
    if "/" in design:
        return os.path.join(bbdir, "examples", design)
    return os.path.join(bbdir, "examples/designs", design)


def design_output_root(bbdir, design, *, selection):
    return os.path.join(owner_root(bbdir, design), "output/sims", selection)


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
        design_output_root(bbdir, design, selection=selection), "emu", backend, "target"
    )


def bebop_bin_path(bbdir, design, backend, *, selection):
    design = _design_name(design)
    if backend not in {"bemu", "verilator", "p2e", "vcs"}:
        raise ValueError(f"unknown Bebop backend: {backend}")
    return os.path.join(
        design_output_root(bbdir, design, selection=selection),
        "bin",
        f"bebop-{backend}",
    )


def design_arch_root(bbdir, design, *, selection):
    return os.path.join(design_output_root(bbdir, design, selection=selection), "arch")


def design_pb_path(bbdir, design, *, selection):
    """The design.pb of one simulation: the design with its config's overrides."""
    return Path(design_output_root(bbdir, design, selection=selection)) / (
        "generated/design.pb"
    )


def load_design_config(bbdir, design, *, selection):
    sys.path.insert(0, str(Path(bbdir) / "bbdev/api/steps/config/scripts"))
    import design_pb2

    path = design_pb_path(bbdir, design, selection=selection)
    return design_pb2.Design.FromString(path.read_bytes())


def simulation_target(bbdir, design, selection):
    return load_design_config(bbdir, design, selection=selection).simulation


def runner_manifest(bbdir, design, selection):
    """The emu crate of the simulation's entry: its model and the runners built from it."""
    return Path(bbdir) / simulation_target(bbdir, design, selection).emu / "Cargo.toml"


def simulation_dir(bbdir, design, *, selection):
    """The entry directory of the simulation: arch, emu, stack, workloads,
    regression and tapeout under it."""
    return Path(bbdir) / simulation_target(bbdir, design, selection).directory


def sim_name(bbdir, design, product, *, selection):
    # The Verilator runner (bebop) always builds against BBSimHarness.
    target = simulation_target(bbdir, design, selection)
    if product == "p2e":
        return "P2ETop"
    return "BBSimHarness" if product == "bebop" else target.top


def rtl_dir(bbdir, design, product, *, selection):
    # Each product (verilog, tapeout, dc, p2e) elaborates its own RTL.
    name = sim_name(bbdir, design, product, selection=selection)
    return os.path.join(
        design_arch_root(bbdir, design, selection=selection), product, name
    )


def rtl_out(bbdir, design, product, *, selection):
    name = sim_name(bbdir, design, product, selection=selection)
    return name, rtl_dir(bbdir, design, product, selection=selection)


def _log_part(value, what):
    if not isinstance(value, str) or not value:
        raise ValueError(f"log dir missing {what}")
    if os.path.basename(value) != value:
        raise ValueError(f"log {what} must be a single path component: {value!r}")
    if not re.fullmatch("[A-Za-z0-9_.-]+", value):
        raise ValueError(f"invalid log {what}: {value!r}")
    return value


def log_dir(bbdir, design, tool, name, *, selection=None):
    """Create a run directory and return its path, reserving the timestamp atomically."""
    # A component alone (its directory relative to examples) is named by its
    # design, or "prefab"; the selection names the component.
    parts = _design_name(design).split("/")
    design = _log_part(parts[1] if parts[0] == "designs" else parts[0], "design")
    tool = _log_part(tool, "tool")
    name = _log_part(os.path.basename(name), "name")
    if selection is not None:
        name = _log_part(selection.replace("/", "-"), "target") + "-" + name
    root = Path(bbdir) / "log"
    root.mkdir(parents=True, exist_ok=True)
    while True:
        now = datetime.now()
        path = root / f"{now:%Y-%m-%d-%H-%M-%S}-{design}-{tool}-{name}"
        try:
            path.mkdir()
            return str(path)
        except FileExistsError:
            # The same design, tool and task are already using this second.
            time.sleep(1 - now.microsecond / 1_000_000)


def run_log_dir(bbdir, run):
    root = (Path(bbdir) / "log").resolve()
    path = (root / _log_part(run, "run")).resolve()
    if path.parent != root:
        raise ValueError(f"run must name a directory under {root}: {run!r}")
    return path
