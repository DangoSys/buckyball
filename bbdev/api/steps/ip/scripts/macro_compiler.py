from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.stream_run import stream_run


def run_macro_compiler(
    *, mems_conf: Path, mdf: Path, verilog: Path, firrtl: Path, arch_dir: Path
) -> None:
    verilog.parent.mkdir(parents=True, exist_ok=True)
    cmd = shlex.join(
        [
            "mill",
            "-i",
            "tapeout.runMain",
            "tapeout.macros.MacroCompiler",
            "-n",
            str(mems_conf),
            "-v",
            str(verilog),
            "-f",
            str(firrtl),
            "--library",
            str(mdf),
            "--mode",
            "strict",
        ]
    )
    r = stream_run(cmd, cwd=str(arch_dir))
    if r.returncode != 0:
        raise RuntimeError(f"MacroCompiler failed:\n{r.stdout}\n{r.stderr}")
    if not verilog.is_file() or verilog.stat().st_size == 0:
        raise RuntimeError(f"MacroCompiler produced empty verilog: {verilog}")
