"""Produce the independent byte oracle at ELF symbol addresses for compute-tile task DMA."""

import argparse
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--nm", required=True)
p.add_argument("--elf", type=Path, required=True)
p.add_argument("--case", choices=("single", "workers"), required=True)
p.add_argument("--workers", type=int, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
symbols = {}
for line in subprocess.check_output(
    [a.nm, "--defined-only", str(a.elf)], text=True
).splitlines():
    fields = line.split()
    if len(fields) == 3:
        symbols[fields[2]] = int(fields[0], 16)


def pattern(seed):
    return bytes((seed * 17 + i * 3) & 255 for i in range(64))


rows = (
    [("task_dma_output", pattern(1))]
    if a.case == "single"
    else [
        ("task_dma_outputs", b"".join(pattern(rank + 1) for rank in range(a.workers))),
        ("task_move_output", pattern(1)),
    ]
)
a.output.write_text("".join(f"{symbols[name]:x} {data.hex()}\n" for name, data in rows))
