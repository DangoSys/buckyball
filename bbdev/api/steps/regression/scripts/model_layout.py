"""Bridge pebble chip layout output -> flat archs/buckyball/<Layout> path the kernel expects.

Workload `--chip <chip> --model <m>` writes under
  bb-tests/output/<chip>/workloads/src/ModelTest/e2e/models/archs/buckyball/<chip>/<Layout>/
Kernel `--model <m>` expects the flat
  bb-tests/output/<chip>/workloads/src/ModelTest/e2e/models/archs/buckyball/<Layout>/
"""
import os
from pathlib import Path

MODEL_LAYOUT = {
    "lenet": "LeNet",
    "mobilenet": "MobileNetV3",
    "resnet": "ResNet18",
    "yolo": "YOLO26",
    "bert": "Bert",
    "qwen3": "Qwen3",
    "gemma4": "Gemma4",
    "deepseekr1": "DeepSeekR1",
    "llama2": "llama2",
    "stable-diffusion": "StableDiffusion",
    "whisper": "Whisper",
    "buddynext": "BuddyNext",
}

MODEL_PERFETTO = {
    "lenet": {
        "trace_toml": "bb-tests/workloads/src/ModelTest/e2e/models/models/LeNet/trace/trace-nodes.toml",
        "mlir_files": [
            "bb-tests/output/{chip}/workloads/src/ModelTest/e2e/models/archs/buckyball/{chip}/LeNet/subgraph0_linalg.mlir",
            "bb-tests/output/{chip}/workloads/src/ModelTest/e2e/models/archs/buckyball/{chip}/LeNet/subgraph0_buckyball.mlir",
        ],
    },
    "mobilenet": {
        "trace_toml": "bb-tests/workloads/src/ModelTest/e2e/models/models/MobileNetV3/trace/trace.toml",
        "mlir_files": [
            "bb-tests/output/{chip}/workloads/src/ModelTest/e2e/models/archs/buckyball/{chip}/MobileNetV3/subgraph0_linalg.mlir",
            "bb-tests/output/{chip}/workloads/src/ModelTest/e2e/models/archs/buckyball/{chip}/MobileNetV3/subgraph0_buckyball.mlir",
        ],
    },
    "resnet": {
        "trace_toml": "bb-tests/workloads/src/ModelTest/e2e/models/models/ResNet18/trace/trace.toml",
        "mlir_files": [
            "bb-tests/output/{chip}/workloads/src/ModelTest/e2e/models/archs/buckyball/{chip}/ResNet18/subgraph0_linalg.mlir",
            "bb-tests/output/{chip}/workloads/src/ModelTest/e2e/models/archs/buckyball/{chip}/ResNet18/subgraph0_buckyball.mlir",
        ],
    },
    "yolo": {
        "trace_toml": "bb-tests/workloads/src/ModelTest/e2e/models/models/YOLO26/trace/trace.toml",
        "mlir_files": [
            "bb-tests/output/{chip}/workloads/src/ModelTest/e2e/models/archs/buckyball/{chip}/YOLO26/subgraph0_linalg.mlir",
            "bb-tests/output/{chip}/workloads/src/ModelTest/e2e/models/archs/buckyball/{chip}/YOLO26/subgraph0_buckyball.mlir",
        ],
    },
}


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
    layout = MODEL_LAYOUT.get(model.lower())
    if layout is None:
        raise ValueError(f"missing layout mapping for model: {model}")
    return layout


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
    spec = MODEL_PERFETTO.get(model.lower())
    if spec is None:
        raise KeyError(f"no perfetto spec for model: {model}")
    trace_toml = Path(bbdir) / spec["trace_toml"]
    if not trace_toml.is_file():
        raise FileNotFoundError(f"perfetto trace toml missing: {trace_toml}")
    mlir_files = []
    for rel in spec["mlir_files"]:
        path = Path(bbdir) / rel.replace("{chip}", chip)
        if not path.is_file():
            raise FileNotFoundError(f"perfetto mlir file missing: {path}")
        mlir_files.append(path)
    return {"trace_toml": trace_toml, "mlir_files": mlir_files}
