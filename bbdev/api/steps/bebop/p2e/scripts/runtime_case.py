from pathlib import Path


def trace_modes(bitstream: str) -> set[str]:
    case = Path(bitstream).resolve().parent.parent
    modes = set((case / "p2e_trace_mode").read_text().strip().split("+"))
    known = {"none", "btrace", "btrace_nb_v1", "itrace", "mtrace"}
    if not modes <= known or ("none" in modes and len(modes) != 1):
        raise ValueError(f"Unknown P2E trace mode: {modes}")
    return modes


def case_ready(bit: Path) -> bool:
    case = bit.parent.parent
    runtime = case / "vvacDir/runtimeDir"
    return all(
        path.is_file()
        for path in (
            bit,
            case / "p2e_trace_mode",
            runtime / "rtcfg",
            runtime / "lib/lib_arm/libvCtb.so",
        )
    )
