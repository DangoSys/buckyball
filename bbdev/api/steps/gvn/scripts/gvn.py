"""GVN, the stimulus path that cuts a model into slices: what the design
drivers share. A design's driver is gvn/run.py of its device simulation
entry, run by `bbdev gvn-bemu --debug`; this module builds the GVN plugins and
runs commands.
"""

import fcntl
import re
import struct
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[5]
# bbdev's own modules (utils.path: log_dir), as its steps import them.
sys.path.insert(0, str(repo / "bbdev/api"))


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


def table_offset(elf):
    """The file offset of gvn_table in an ELF."""
    image = elf.read_bytes()
    symbols = subprocess.check_output(
        [repo / "stack/result/bin/riscv64-unknown-elf-nm", "--defined-only", elf],
        text=True,
    )
    (address,) = re.findall(r"^([0-9a-fA-F]+) \w gvn_table$", symbols, re.MULTILINE)
    address = int(address, 16)
    phoff = struct.unpack_from("<Q", image, 32)[0]
    phentsize, phnum = struct.unpack_from("<HH", image, 54)
    segments = [
        struct.unpack_from("<IIQQQQQQ", image, phoff + i * phentsize)
        for i in range(phnum)
    ]
    (segment,) = [
        segment
        for segment in segments
        if segment[0] == 1 and segment[3] <= address < segment[3] + segment[5]
    ]
    return segment[2] + address - segment[3]
