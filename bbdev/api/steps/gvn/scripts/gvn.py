"""GVN, the stimulus path that cuts a model into slices: what the design
drivers share. A design's driver is gvn/run.py of its device simulation
entry, run by `bbdev gvn-bemu --debug`; this module builds the GVN plugins and
runs commands.
"""

import asyncio
import fcntl
import functools
import re
import struct
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[5]
# bbdev's own modules (utils.path: log_dir), as its steps import them.
sys.path.insert(0, str(repo / "bbdev/api"))

from utils.path import design_output_root  # noqa: E402


def model_turn(handler):
    """Protect the model's generated MLIR, slice ELFs and manifest across workers."""

    @functools.wraps(handler)
    async def locked(data, ctx):
        if "model" not in data:
            return await handler(data, ctx)
        output = Path(
            design_output_root(repo, data["design"], selection=data["target"])
        )
        directory = output / "gvn"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f".{Path(data['model']).name}.lock").open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(1)
            return await handler(data, ctx)

    return locked


def run(command, **options):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, **options)


def build_plugins():
    """Build stack/gvn: the gvn plugin, and the generic midend and the nv
    passes of the GPU side. Runs of several models build it in turn."""
    build = repo / "stack/gvn/build"
    build.mkdir(parents=True, exist_ok=True)
    with (build / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(
            ["cmake", "-G", "Ninja", "-S", repo / "stack/gvn", "-B", build],
            stdout=subprocess.DEVNULL,
        )
        run(["ninja", "-C", build])
    return (
        build / "gvn-compiler/libgvn-compiler.so",
        build / "generic-compiler/libgeneric-compiler.so",
        build / "nv-compiler/libnv-compiler.so",
    )


def bank_operations(text):
    """The physical bank operations of a graph, without values and locations.
    Bank allocs and releases are left out: assign-physical-banks keeps a bank
    held across the end of one subgraph and the start of the next, so a slice
    compiled alone allocates and releases where the whole graph does not."""
    pattern = re.compile(
        r'^\s*(?:%[^=]+\s*=\s*)?"?((?:bbacc|\w+_ball)\.\w+)"?[^<\n]*(<[^>]*>)?'
    )
    operations = []
    for line in text.splitlines():
        match = pattern.search(line)
        if match and match.group(1) != "bbacc.mset":
            operations.append(match.group(1) + (match.group(2) or ""))
    return operations


def symbol_offset(elf, symbol):
    """Locate a file-backed RV64 word in the loadable ELF segments."""
    symbols = subprocess.check_output(
        [repo / "stack/result/bin/riscv64-unknown-elf-nm", "--defined-only", elf],
        text=True,
    )
    (address,) = re.findall(rf"^([0-9a-fA-F]+) \w {re.escape(symbol)}$", symbols, re.M)
    address = int(address, 16)
    with elf.open("rb") as stream:
        header = struct.unpack("<16sHHIQQQIHHHHHH", stream.read(64))
        stream.seek(header[5])
        segments = [
            struct.unpack_from("<IIQQQQQQ", stream.read(header[9]))
            for _ in range(header[10])
        ]
    (segment,) = [
        s
        for s in segments
        if s[0] == 1 and s[3] <= address and address + 8 <= s[3] + s[5]
    ]
    return segment[2] + address - segment[3]


def table_offset(elf):
    return symbol_offset(elf, "gvn_table")
