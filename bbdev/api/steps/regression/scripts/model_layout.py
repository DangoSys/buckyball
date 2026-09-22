"""Bridge pebble chip layout output -> flat archs/buckyball/<Layout> path the kernel expects.

Workload `--chip <chip> --model <m>` writes under
  bb-tests/output/<chip>/workloads/src/ModelTest/e2e/models/archs/buckyball/<chip>/<Layout>/
Kernel `--model <m>` expects the flat
  bb-tests/output/<chip>/workloads/src/ModelTest/e2e/models/archs/buckyball/<Layout>/
"""
import os
from pathlib import Path
from utils.model import model_layout_name


def _archs_root(bbdir: str, chip: str) -> Path:
    return (
        Path(bbdir)
        / "bb-tests"
        / "output"
        / chip
        / "workloads"
        / "src"
        / "ModelTest"
        / "e2e"
        / "models"
        / "archs"
        / "buckyball"
    )


def layout_name(model: str) -> str:
    return model_layout_name(model)


def chip_output_dir(bbdir: str, chip: str, model: str) -> Path:
    return _archs_root(bbdir, chip) / chip / layout_name(model)


def flat_output_dir(bbdir: str, chip: str, model: str) -> Path:
    return _archs_root(bbdir, chip) / layout_name(model)


def bridge_model_layout(bbdir: str, chip: str, model: str) -> Path:
    source = chip_output_dir(bbdir, chip, model)
    if not source.is_dir():
        raise FileNotFoundError(
            f"chip layout output dir missing for model '{model}' on chip '{chip}': {source}"
        )
    flat = flat_output_dir(bbdir, chip, model)
    if flat.is_symlink():
        target = os.readlink(flat)
        if os.path.abspath(target) == os.path.abspath(str(source)):
            return flat
        flat.unlink()
    elif flat.exists():
        raise FileExistsError(
            f"refusing to clobber existing non-symlink at flat layout path: {flat}"
        )
    flat.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(source), str(flat))
    return flat


def perfetto_inputs(bbdir: str, chip: str, model: str) -> dict:
    if model.lower() not in {"lenet", "mobilenet", "resnet", "yolo"}:
        raise KeyError(f"no perfetto spec for model: {model}")
    layout = layout_name(model)
    trace_name = "trace-nodes.toml" if model.lower() == "lenet" else "trace.toml"
    trace_toml = (
        Path(bbdir)
        / "bb-tests/workloads/src/ModelTest/e2e/models/models"
        / layout
        / "trace"
        / trace_name
    )
    if not trace_toml.is_file():
        raise FileNotFoundError(f"perfetto trace toml missing: {trace_toml}")
    output = _archs_root(bbdir, chip) / chip / layout
    mlir_files = []
    for name in ("subgraph0_linalg.mlir", "subgraph0_buckyball.mlir"):
        path = output / name
        if not path.is_file():
            raise FileNotFoundError(f"perfetto mlir file missing: {path}")
        mlir_files.append(path)
    return {"trace_toml": trace_toml, "mlir_files": mlir_files}
