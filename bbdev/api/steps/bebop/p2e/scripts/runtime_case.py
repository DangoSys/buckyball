from pathlib import Path
import os


def validate_runtime_reuse(bitstream: str, diff: bool) -> Path:
    bit = Path(bitstream).resolve()
    case = bit.parent.parent
    executable = case / "bebop-p2e"
    files = [bit, executable, case / "vvacDir/runtimeDir/rtcfg",
             case / "vvacDir/runtimeDir/lib/lib_arm/libvCtb.so",
             case / "p2e_trace_mode"]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise ValueError(f"Reusable P2E case is incomplete: {missing}")
    if not os.access(executable, os.X_OK):
        raise ValueError(f"Reusable P2E runtime is not executable: {executable}")
    mode = (case / "p2e_trace_mode").read_text().strip()
    if mode not in ("none", "btrace", "btrace_nb_v1"):
        raise ValueError(f"Unknown reusable P2E trace mode: {mode}")
    if diff and mode == "none":
        raise ValueError("Reusable P2E case does not support diff")
    return executable
