from pathlib import Path
import os


def trace_modes(bitstream: str) -> set[str]:
    case = Path(bitstream).resolve().parent.parent
    modes = set((case / "p2e_trace_mode").read_text().strip().split("+"))
    known = {"none", "btrace", "btrace_nb_v1", "itrace", "mtrace"}
    if not modes <= known or ("none" in modes and len(modes) != 1):
        raise ValueError(f"Unknown P2E trace mode: {modes}")
    return modes


def validate_runtime_reuse(bitstream: str, diff: bool, itrace: bool = False, mtrace: bool = False) -> Path:
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
    modes = trace_modes(bitstream)
    if diff and not modes & {"btrace", "btrace_nb_v1"}:
        raise ValueError("Reusable P2E case does not support diff")
    for trace, requested in (("itrace", itrace), ("mtrace", mtrace)):
        if requested and trace not in modes:
            raise ValueError(f"Reusable P2E case does not support {trace}; rebuild with --{trace}")
    return executable
