#!/usr/bin/env python3
"""Install Chip.pb into arch / bemu / compiler / workload build trees."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import chip_pb2 as pb  # noqa: E402

_ENGINE = Path("bebop/src/nodes/bemu")
_CHIP_RS_DIR = _ENGINE / "src"
_BEMU_CRATE = _ENGINE / "chip"


def load_chip(pb_path: Path) -> pb.Chip:
    chip = pb.Chip()
    chip.ParseFromString(pb_path.read_bytes())
    if not chip.name:
        raise ValueError(f"empty Chip.name in {pb_path}")
    if not chip.cores:
        raise ValueError(f"no cores in {pb_path}")
    return chip


def _emit_dispatch(chip: pb.Chip, bbdir: Path, out: Path) -> None:
    balls = list(chip.bemu.balls)
    src_dir = bbdir / _CHIP_RS_DIR
    if not balls:
        out.write_text(
            "use crate::inst::instruction::ExecContext;\n\n"
            "pub fn execute_known(\n"
            "    _ball_class: &str,\n"
            "    _funct: u32,\n"
            "    _xs1: u64,\n"
            "    _xs2: u64,\n"
            "    _ctx: &mut ExecContext,\n"
            ") -> u64 {\n"
            '    panic!("no BEMU ball implementation")\n'
            "}\n\n"
            "pub fn cycles_after_issue(_ball_class: &str, _funct: u32, _xs1: u64, _xs2: u64) -> u64 {\n"
            '    panic!("no BEMU ball latency implementation")\n'
            "}\n",
            encoding="utf-8",
        )
        return

    lines = ["use crate::inst::instruction::ExecContext;", ""]
    for ball in balls:
        if not ball.ball_dir.isidentifier():
            raise ValueError(f"ball_dir is not a Rust identifier: {ball.ball_dir!r}")
        lib = bbdir / ball.emu_lib
        if not lib.is_file():
            raise FileNotFoundError(f"missing ball emu: {lib}")
        rel = Path(os.path.relpath(lib, src_dir)).as_posix()
        if '"' in rel or "\\" in rel:
            raise ValueError(f"ball emu path not usable in rust #[path]: {rel}")
        lines.append(f'#[path = "{rel}"]')
        lines.append(f"mod {ball.ball_dir};")
        lines.append("")

    def chain(fn: str, ctx: bool, panic_msg: str) -> list[str]:
        extra = ", ctx" if ctx else ""
        body = [
            f"    {balls[0].ball_dir}::{fn}(ball_class, funct, xs1, xs2{extra})"
        ]
        for ball in balls[1:]:
            body.append(
                f"        .or_else(|| {ball.ball_dir}::{fn}(ball_class, funct, xs1, xs2{extra}))"
            )
        body.append(f'        .unwrap_or_else(|| panic!("{panic_msg}"))')
        return body

    lines += [
        "pub fn execute_known(",
        "    ball_class: &str,",
        "    funct: u32,",
        "    xs1: u64,",
        "    xs2: u64,",
        "    ctx: &mut ExecContext,",
        ") -> u64 {",
    ]
    lines += chain(
        "execute_known",
        True,
        "no BEMU ball implementation for ballClass={ball_class} funct7={funct}",
    )
    lines += ["}", ""]
    lines += [
        "pub fn cycles_after_issue(ball_class: &str, funct: u32, xs1: u64, xs2: u64) -> u64 {",
    ]
    lines += chain(
        "cycles_after_issue",
        False,
        "no BEMU ball latency implementation for ballClass={ball_class} funct7={funct}",
    )
    lines += ["}", ""]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def install_arch(src_pb: Path, gen: Path) -> Path:
    dest = gen / "chip.pb"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_pb, dest)
    return dest


def install_bemu(chip: pb.Chip, bbdir: Path, gen: Path) -> Path:
    src = bbdir / _BEMU_CRATE
    cargo = src / "Cargo.toml"
    build_rs = src / "build.rs"
    if not cargo.is_file():
        raise FileNotFoundError(f"missing {cargo}")
    if not build_rs.is_file():
        raise FileNotFoundError(f"missing {build_rs}")
    bemu = gen / "bemu"
    bemu.mkdir(parents=True, exist_ok=True)
    _emit_dispatch(chip, bbdir, bemu / "dispatch.rs")
    shutil.copy2(cargo, bemu / "Cargo.toml")
    shutil.copy2(build_rs, bemu / "build.rs")
    return bemu / "Cargo.toml"


def install_workload(chip: pb.Chip, bbdir: Path, name: str, gen: Path) -> Path:
    if not chip.profiles:
        raise ValueError(f"chip {name}: no compiler profiles")
    cargo = bbdir / "bebop" / "target" / name
    defs = {
        "BUCKYBALL_WORKLOAD_CHIP": name,
        "BUCKYBALL_CARGO_TARGET_DIR": str(cargo),
        "BUCKYBALL_RUSHB_BEMU_MANIFEST": str(gen / "bemu" / "Cargo.toml"),
        "BUCKYBALL_RUSHB_BEMU_LIBRARY": str(cargo / "release" / "libbebop_bemu.so"),
        "BUCKYBALL_RUSHB_VERILATOR_LIBRARY": str(
            cargo / "release" / "deps" / "libbebop_verilator.so"
        ),
        "BUCKYBALL_CHIP_PB": str(gen / "chip.pb"),
    }
    out = gen / "workload" / "cmake.defs"
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{k}={v}\n" for k, v in sorted(defs.items()))
    out.write_text(body, encoding="utf-8")
    return out
