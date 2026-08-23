from __future__ import annotations

import os
import shlex
from pathlib import Path

from utils.chip import chip_field
from utils.path import chip_arch_root
from utils.stream_run import stream_run_logger_async

_KIND = {
    "verilog": ("verilatorConfig", "sims.verilator.Elaborate", False, "fesvr"),
    "synth": ("verilatorConfig", "sims.verilator.Elaborate", True, None),
    "p2e": ("p2eConfig", "sims.p2e.Elaborate", False, "p2e"),
}


def mill_cmd(main: str, mill_config: str, out: str, *, seq_mem: bool = False) -> str:
    extra = ""
    if seq_mem:
        extra = (
            " -lowering-options=disallowLocalVariables"
            f" --repl-seq-mem --repl-seq-mem-file={shlex.quote(os.path.join(out, 'mems.conf'))}"
        )
    return (
        f"mill -i __.test.runMain {main} {mill_config} "
        "--disable-annotation-unknown --strip-debug-info -O=debug "
        f"--split-verilog -o={shlex.quote(out)}{extra}"
    )


def p2e_top_cmd(out: str) -> str:
    return (
        "mill -i __.test.runMain sims.p2e.ElaborateP2ETop "
        "--disable-annotation-unknown --strip-debug-info -O=debug "
        f"--split-verilog -o={shlex.quote(out)}"
    )


def patch_fesvr(out: str, arch: str) -> None:
    harness = os.path.join(arch, "BBSimHarness.sv")
    if os.path.isfile(harness):
        os.remove(harness)
    drop = ("fesvr/memif.h", "fesvr/elfloader.h")
    for name in ("mm.h", "mm.cc"):
        path = os.path.join(out, name)
        if not os.path.isfile(path):
            continue
        text = Path(path).read_text()
        Path(path).write_text(
            "".join(line for line in text.splitlines(True) if not any(token in line for token in drop))
        )


def drop_p2e_harness(arch: str) -> None:
    for name in ("P2EHarness.sv", "P2ETop.v", "P2ETopWrapper.sv"):
        path = os.path.join(arch, name)
        if os.path.isfile(path):
            os.remove(path)


def normalize_timescale(out: str) -> None:
    n = 0
    for path in Path(out).rglob("*"):
        if path.suffix not in {".v", ".sv"}:
            continue
        text = path.read_text()
        if "`timescale" in text:
            continue
        path.write_text("`timescale 1ns/1ps\n" + text)
        n += 1
    return n


def assert_sv(out: str) -> None:
    n = sum(1 for path in Path(out).rglob("*") if path.suffix in {".v", ".sv"})
    if n == 0:
        raise RuntimeError(f"no .sv/.v emitted under {out}")


async def run_chip_mill(ctx, bbdir: str, chip: str, kind: str, prefix: str, output_dir=None):
    if kind not in _KIND:
        raise ValueError(f"invalid mill kind: {kind}")
    arch = os.path.join(bbdir, "arch")
    if not os.path.isdir(arch):
        raise ValueError(f"missing {arch}")
    key, main, seq_mem, post = _KIND[kind]
    mill_config = chip_field(bbdir, chip, key)
    out = output_dir or os.path.join(chip_arch_root(bbdir, chip), mill_config)
    os.makedirs(out, exist_ok=True)
    ctx.logger.info(f"Using mill config: {mill_config}")
    ctx.logger.info(f"Using build directory: {out}")
    result = await stream_run_logger_async(
        cmd=mill_cmd(main, mill_config, out, seq_mem=seq_mem),
        logger=ctx.logger,
        cwd=arch,
        stdout_prefix=prefix,
        stderr_prefix=prefix,
    )
    if result.returncode != 0:
        return out, result.returncode
    if post == "fesvr":
        patch_fesvr(out, arch)
    if post == "p2e":
        top = await stream_run_logger_async(
            cmd=p2e_top_cmd(out),
            logger=ctx.logger,
            cwd=arch,
            stdout_prefix=prefix,
            stderr_prefix=prefix,
        )
        if top.returncode != 0:
            return out, top.returncode
        drop_p2e_harness(arch)
        n = normalize_timescale(out)
        ctx.logger.info(f"normalized timescale in {n} files")
    if seq_mem:
        mems = os.path.join(out, "mems.conf")
        if not os.path.isfile(mems):
            raise RuntimeError(f"mill seq-mem did not emit {mems}")
    assert_sv(out)
    return out, 0
