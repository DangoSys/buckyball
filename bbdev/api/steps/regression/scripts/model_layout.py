from pathlib import Path
from utils.model import model_layout_name


def perfetto_inputs(bbdir: str, chip: str, model: str) -> dict:
    name = model_layout_name(model)
    root = Path(bbdir)
    trace_toml = root / "examples/models" / chip / name / "trace/trace.toml"
    mlir = root / "stack/models/build" / chip / name / "generated/subgraph0.mlir"
    for path in (trace_toml, mlir):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {"trace_toml": trace_toml, "mlir_files": [mlir]}
