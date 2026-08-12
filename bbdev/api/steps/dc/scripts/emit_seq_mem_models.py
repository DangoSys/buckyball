#!/usr/bin/env python3
"""Emit behavioral models for FIRRTL externalized sequential memories.

These models are used only by gate-level activity simulation when a chip has
not yet supplied PDK SRAM macro simulation collateral.  They retain the
external-memory interface from ``mems.conf`` so the DC netlist is simulated
with functional memories rather than X-producing black boxes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LINE = re.compile(
    r"name\s+(?P<name>\S+)\s+depth\s+(?P<depth>\d+)\s+width\s+(?P<width>\d+)"
    r"\s+ports\s+(?P<ports>\S+)(?:\s+mask_gran\s+(?P<mask>\d+))?$"
)


def write_masked(lines: list[str], *, addr: str, data: str, mask: str, width: int, gran: int) -> None:
    lines.extend(
        [
            f"      for (int i = 0; i < {width // gran}; i++) begin",
            f"        if ({mask}[i]) memory[{addr}][i*{gran} +: {gran}] <= {data}[i*{gran} +: {gran}];",
            "      end",
        ]
    )


def emit(name: str, depth: int, width: int, ports: str, mask_gran: int | None) -> str:
    lines = [f"module {name}("]
    if ports == "mrw":
        lines += [
            "  input logic [$clog2(" + str(depth) + ")-1:0] RW0_addr,",
            "  input logic RW0_en, RW0_clk, RW0_wmode,",
            f"  input logic [{width - 1}:0] RW0_wdata,",
            f"  output logic [{width - 1}:0] RW0_rdata,",
            f"  input logic [{width // mask_gran - 1}:0] RW0_wmask",
            ");",
            f"  logic [{width - 1}:0] memory [0:{depth - 1}];",
            "  always @(posedge RW0_clk) begin",
            "    if (RW0_en) begin",
            "      if (RW0_wmode) begin",
        ]
        write_masked(lines, addr="RW0_addr", data="RW0_wdata", mask="RW0_wmask", width=width, gran=mask_gran or width)
        lines += ["      end", "      RW0_rdata <= memory[RW0_addr];", "    end", "  end"]
    elif ports == "rw":
        lines += [
            "  input logic [$clog2(" + str(depth) + ")-1:0] RW0_addr,",
            "  input logic RW0_en, RW0_clk, RW0_wmode,",
            f"  input logic [{width - 1}:0] RW0_wdata,",
            f"  output logic [{width - 1}:0] RW0_rdata",
            ");",
            f"  logic [{width - 1}:0] memory [0:{depth - 1}];",
            "  always @(posedge RW0_clk) begin",
            "    if (RW0_en) begin",
            "      if (RW0_wmode) memory[RW0_addr] <= RW0_wdata;",
            "      RW0_rdata <= memory[RW0_addr];",
            "    end",
            "  end",
        ]
    elif ports in {"mwrite,read", "write,read"}:
        has_mask = ports.startswith("mwrite")
        lines += [
            "  input logic [$clog2(" + str(depth) + ")-1:0] R0_addr,",
            "  input logic R0_en, R0_clk,",
            f"  output logic [{width - 1}:0] R0_data,",
            "  input logic [$clog2(" + str(depth) + ")-1:0] W0_addr,",
            "  input logic W0_en, W0_clk,",
            f"  input logic [{width - 1}:0] W0_data" + ("," if has_mask else ""),
        ]
        if has_mask:
            lines.append(f"  input logic [{width // mask_gran - 1}:0] W0_mask")
        lines += [");", f"  logic [{width - 1}:0] memory [0:{depth - 1}];", "  always @(posedge R0_clk) if (R0_en) R0_data <= memory[R0_addr];", "  always @(posedge W0_clk) if (W0_en) begin"]
        if has_mask:
            write_masked(lines, addr="W0_addr", data="W0_data", mask="W0_mask", width=width, gran=mask_gran or width)
        else:
            lines.append("    memory[W0_addr] <= W0_data;")
        lines += ["  end"]
    else:
        raise ValueError(f"unsupported memory port topology for {name}: {ports}")
    lines += ["endmodule", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem-conf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = []
    for raw in args.mem_conf.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid mems.conf entry: {line}")
        values = match.groupdict()
        models.append(
            emit(
                values["name"],
                int(values["depth"]),
                int(values["width"]),
                values["ports"],
                int(values["mask"]) if values["mask"] else None,
            )
        )
    if not models:
        raise ValueError(f"no memories found in {args.mem_conf}")
    args.output.write_text("`timescale 1ns/1ps\n\n" + "\n".join(models), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
