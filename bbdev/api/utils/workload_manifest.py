from pathlib import Path
from .path import get_buckyball_path, simulation_target


def design_regression_dir(
    design: str, backend: str, bbdir: str | None = None, *, selection: str
) -> Path:
    root = Path(bbdir or get_buckyball_path())
    return root / simulation_target(root, design, selection).regression / backend


def resolve_workload_toml(
    design: str,
    backend: str,
    test_type: str,
    bbdir: str | None = None,
    diff: bool = False,
    *,
    selection: str,
) -> str:
    if test_type == "bare-tests":
        suffix = "bare"
    elif test_type == "linux-tests":
        suffix = "linux"
    elif test_type == "slice-tests":
        suffix = "slices"
    else:
        raise ValueError(f"invalid test type: {test_type}")
    regression_dir = design_regression_dir(design, backend, bbdir, selection=selection)
    if diff:
        diff_toml = regression_dir / f"workloads-{suffix}-diff.toml"
        if diff_toml.is_file():
            return str(diff_toml)
    name = f"workloads-{suffix}.toml"
    toml = regression_dir / name
    if not toml.is_file():
        raise ValueError(f"regression workload toml does not exist: {toml}")
    return str(toml)
