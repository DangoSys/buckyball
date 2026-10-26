import os
from pathlib import Path
from .path import get_buckyball_path, design_resource_dir


def validate_design(design: str | None) -> str:
    if not design:
        raise ValueError("missing required parameter: design")
    if not isinstance(design, str):
        raise ValueError("design must be a string")
    if os.path.sep in design or design in {".", ".."}:
        raise ValueError(f"invalid design: {design}")
    return design


def design_regression_dir(
    design: str, backend: str, bbdir: str | None = None, *, selection: str
) -> Path:
    if selection != "device":
        raise ValueError(
            "batch regression requires --device; use a single workload for --core"
        )
    design = validate_design(design)
    root = Path(bbdir or get_buckyball_path())
    return (
        design_resource_dir(root, design, "regression", selection=selection) / backend
    )


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
