"""Generate SRAM replacement collateral for behavioral memories.

The generated artifacts mirror the DC tapeout flow: behavioral memory modules
are replaced by same-named wrappers around compiler macros, while the original
RTL is removed from the synthesis source list.  Unsupported port topologies
remain in RTL and are recorded in the manifest instead of being approximated.
"""

from __future__ import annotations

import json
import math
import os
import re
import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path


MODULE_RE = re.compile(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\s*\((.*?)\)\s*;", re.S)
MEMORY_RE = re.compile(
    r"\breg\s+(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s+)?Memory\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]",
    re.S,
)
SIZE_RE = re.compile(r"\bSIZE\s+([0-9.]+)\s+BY\s+([0-9.]+)\s*;", re.I)
RANGE_RE = re.compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*{name}\b")


@dataclass(frozen=True)
class Macro:
    name: str
    depth: int
    width: int
    write_size: int
    area: float
    liberty: Path
    verilog: Path
    lef: Path


@dataclass(frozen=True)
class Memory:
    name: str
    source: Path
    header: str
    depth: int
    width: int
    addr_width: int
    mask_width: int | None


def width_from_range(text: str, name: str) -> int | None:
    match = re.search(RANGE_RE.pattern.format(name=re.escape(name)), text)
    if match is None:
        return None
    return abs(int(match.group(1)) - int(match.group(2))) + 1


def ceil_log2(value: int) -> int:
    return max(1, math.ceil(math.log2(value)))


def load_macros(sky130_root: Path) -> list[Macro]:
    macro_root = sky130_root / "share" / "sram22_sky130_macros"
    macros: list[Macro] = []
    for toml_path in sorted(macro_root.glob("*/sram22.toml")):
        name = toml_path.parent.name
        if name.endswith("_test"):
            continue
        with toml_path.open("rb") as handle:
            config = tomllib.load(handle)
        liberty = toml_path.parent / f"{name}_tt_025C_1v80.rc.lib"
        verilog = toml_path.parent / f"{name}.v"
        lef = toml_path.parent / f"{name}.lef"
        if not (liberty.is_file() and verilog.is_file() and lef.is_file()):
            continue
        size = SIZE_RE.search(lef.read_text(errors="replace"))
        if size is None:
            continue
        macros.append(
            Macro(
                name=name,
                depth=int(config["num_words"]),
                width=int(config["data_width"]),
                write_size=int(config["write_size"]),
                area=float(size.group(1)) * float(size.group(2)),
                liberty=liberty,
                verilog=verilog,
                lef=lef,
            )
        )
    if not macros:
        raise RuntimeError(f"no usable SRAM22 macros found under {macro_root}")
    return macros


def discover_memories(source_paths: list[Path]) -> tuple[list[Memory], list[dict[str, object]]]:
    memories: list[Memory] = []
    unsupported: list[dict[str, object]] = []
    for source in source_paths:
        text = source.read_text(errors="replace")
        module_match = MODULE_RE.search(text)
        memory_match = MEMORY_RE.search(text)
        if module_match is None or memory_match is None:
            continue
        name, header = module_match.groups()
        ports = set(re.findall(r"\b(?:RW0|R0|W0)_[A-Za-z0-9_]+\b", header))
        if not {"RW0_addr", "RW0_en", "RW0_clk", "RW0_wmode", "RW0_wdata", "RW0_rdata"}.issubset(ports):
            unsupported.append(
                {
                    "module": name,
                    "source": str(source),
                    "status": "unsupported",
                    "reason": "requires a multi-port SRAM macro; SRAM22 provides only 1RW macros",
                }
            )
            continue
        data_msb = int(memory_match.group(1) or 0)
        data_lsb = int(memory_match.group(2) or 0)
        depth_a = int(memory_match.group(3))
        depth_b = int(memory_match.group(4))
        addr_width = width_from_range(header, "RW0_addr")
        if addr_width is None:
            addr_width = ceil_log2(abs(depth_a - depth_b) + 1)
        memories.append(
            Memory(
                name=name,
                source=source,
                header=header,
                depth=abs(depth_a - depth_b) + 1,
                width=abs(data_msb - data_lsb) + 1,
                addr_width=addr_width,
                mask_width=width_from_range(header, "RW0_wmask"),
            )
        )
    return memories, unsupported


def reachable_modules(source_paths: list[Path], top_module: str | None) -> set[str] | None:
    if not top_module:
        return None
    modules: dict[str, str] = {}
    for source in source_paths:
        text = source.read_text(errors="replace")
        match = MODULE_RE.search(text)
        if match is not None:
            modules[match.group(1)] = text
    if top_module not in modules:
        return {top_module}

    # Split Verilog module instantiations are line-oriented in CIRCT output.
    # The pattern intentionally requires a module type and an instance name,
    # excluding ``module``, ``always`` and other language constructs.
    instance_re = re.compile(
        r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s*(?:#\s*\(.*?\))?\s+([A-Za-z_][A-Za-z0-9_$]*)\s*\(",
        re.M | re.S,
    )
    reachable = {top_module}
    pending = [top_module]
    while pending:
        module = pending.pop()
        for child, _ in instance_re.findall(modules.get(module, "")):
            if child in modules and child not in reachable:
                reachable.add(child)
                pending.append(child)
    return reachable


def choose_macro(memory: Memory, macros: list[Macro]) -> Macro:
    source_granularity = memory.width // memory.mask_width if memory.mask_width else None
    candidates = [
        macro
        for macro in macros
        if source_granularity is None or source_granularity % macro.write_size == 0
    ]
    if not candidates:
        raise RuntimeError(f"no SRAM22 macro has a compatible write mask for {memory.name}")
    return min(
        candidates,
        key=lambda macro: (
            math.ceil(memory.depth / macro.depth)
            * math.ceil(memory.width / macro.width)
            * macro.area,
            -macro.width,
        ),
    )


def addr_expr(signal: str, source_width: int, macro_width: int) -> str:
    if source_width == macro_width:
        return signal
    if source_width > macro_width:
        return f"{signal}[{macro_width - 1}:0]"
    return f"{{{{{macro_width - source_width}{{1'b0}}}}, {signal}}}"


def data_expr(signal: str, lo: int, logical_width: int, physical_width: int) -> str:
    if logical_width == physical_width:
        return f"{signal}[{lo} +: {logical_width}]"
    return f"{{{{{physical_width - logical_width}{{1'b0}}}}, {signal}[{lo} +: {logical_width}]}}"


def bank_select_expr(memory: Memory, macro: Macro, bank: int, banks: int) -> str:
    if banks == 1:
        return "1'b1"
    macro_addr_width = ceil_log2(macro.depth)
    bank_bits = ceil_log2(banks)
    return f"(RW0_addr[{macro_addr_width + bank_bits - 1}:{macro_addr_width}] == {bank_bits}'d{bank})"


def emit_wrapper(memory: Memory, macro: Macro) -> tuple[str, dict[str, object]]:
    macro_addr_width = ceil_log2(macro.depth)
    banks = math.ceil(memory.depth / macro.depth)
    bank_bits = ceil_log2(banks) if banks > 1 else 0
    slices = math.ceil(memory.width / macro.width)
    lines = [f"module {memory.name}({memory.header});", ""]
    lines.extend(
        [
            "  logic ren_d0;",
            "  logic wmode_d0;",
            f"  logic [{memory.width - 1}:0] rdata_comb;",
        ]
    )
    if bank_bits:
        lines.append(f"  logic [{bank_bits - 1}:0] bank_d0;")
    lines.extend(["", "  always_ff @(posedge RW0_clk) begin", "    ren_d0 <= RW0_en;", "    wmode_d0 <= RW0_wmode;"])
    if bank_bits:
        lines.append(f"    bank_d0 <= RW0_addr[{macro_addr_width + bank_bits - 1}:{macro_addr_width}];")
    lines.extend(["  end", ""])

    for slice_index in range(slices):
        lo = slice_index * macro.width
        logical_width = min(macro.width, memory.width - lo)
        lines.append(f"  logic [{macro.width - 1}:0] q_{slice_index} [0:{banks - 1}];")
        lines.append(f"  logic [{macro.width - 1}:0] d_{slice_index};")
        lines.append(f"  logic [{macro.width // macro.write_size - 1}:0] wmask_{slice_index};")
        lines.append(f"  assign d_{slice_index} = {data_expr('RW0_wdata', lo, logical_width, macro.width)};")
        for bit in range(macro.width // macro.write_size):
            if memory.mask_width is None:
                mask = "1'b1"
            else:
                source_granularity = memory.width // memory.mask_width
                source_bit = (lo + bit * macro.write_size) // source_granularity
                mask = f"RW0_wmask[{source_bit}]" if source_bit < memory.mask_width else "1'b0"
            lines.append(f"  assign wmask_{slice_index}[{bit}] = {mask};")
        lines.append("")
        for bank in range(banks):
            select = bank_select_expr(memory, macro, bank, banks)
            lines.extend(
                [
                    f"  {macro.name} u_sram_{slice_index}_{bank} (",
                    "    .clk(RW0_clk),",
                    f"    .we(RW0_en & RW0_wmode & {select}),",
                    f"    .wmask(wmask_{slice_index}),",
                    f"    .addr({addr_expr('RW0_addr', memory.addr_width, macro_addr_width)}),",
                    f"    .din(d_{slice_index}),",
                    f"    .dout(q_{slice_index}[{bank}])",
                    "  );",
                ]
            )
        lines.append("")

    lines.append("  always_comb begin")
    lines.append("    rdata_comb = 'x;")
    for slice_index in range(slices):
        lo = slice_index * macro.width
        logical_width = min(macro.width, memory.width - lo)
        if banks == 1:
            lines.append(f"    rdata_comb[{lo} +: {logical_width}] = q_{slice_index}[0][{logical_width - 1}:0];")
        else:
            lines.append("    case (bank_d0)")
            for bank in range(banks):
                lines.append(
                    f"      {bank_bits}'d{bank}: rdata_comb[{lo} +: {logical_width}] = q_{slice_index}[{bank}][{logical_width - 1}:0];"
                )
            lines.append("      default: rdata_comb = 'x;")
            lines.append("    endcase")
    lines.extend(
        [
            "  end",
            f"  assign RW0_rdata = (ren_d0 && !wmode_d0) ? rdata_comb : {memory.width}'bx;",
            "endmodule",
            "",
        ]
    )
    return "\n".join(lines), {
        "module": memory.name,
        "source": str(memory.source),
        "status": "mapped",
        "port_type": "1RW",
        "logical_depth": memory.depth,
        "logical_width": memory.width,
        "macro": macro.name,
        "macro_depth": macro.depth,
        "macro_width": macro.width,
        "macro_write_size": macro.write_size,
        "banks": banks,
        "slices": slices,
        "instances": banks * slices,
        "macro_area": macro.area,
        "macro_total_area": macro.area * banks * slices,
        "liberty": str(macro.liberty),
        "lef": str(macro.lef),
        "verilog": str(macro.verilog),
    }


def emit_blackboxes(macros: list[Macro]) -> str:
    lines = ["// SRAM22 blackboxes for synthesis; physical views are in manifest.", ""]
    for macro in sorted({macro.name: macro for macro in macros}.values(), key=lambda item: item.name):
        addr_width = ceil_log2(macro.depth)
        mask_width = macro.width // macro.write_size
        lines.extend(
            [
                "(* blackbox *)",
                f"module {macro.name}(",
                "  input clk,",
                "  input we,",
                f"  input [{mask_width - 1}:0] wmask,",
                f"  input [{addr_width - 1}:0] addr,",
                f"  input [{macro.width - 1}:0] din,",
                f"  output [{macro.width - 1}:0] dout",
                ");",
                "endmodule",
                "",
            ]
        )
    return "\n".join(lines)


def prepare_ip_replacement(
    source_paths: list[str], output_dir: str, sky130_root: str | None, top_module: str | None = None
) -> dict[str, object]:
    """Create IP replacement collateral for DC and Yosys consumers."""
    if not sky130_root:
        return {"source_paths": source_paths, "macro_liberties": [], "macro_area": 0.0, "mapped": 0}
    root = Path(os.path.expandvars(os.path.expanduser(sky130_root)))
    if not root.is_dir():
        return {"source_paths": source_paths, "macro_liberties": [], "macro_area": 0.0, "mapped": 0}

    macros = load_macros(root)
    sources = [Path(path) for path in source_paths]
    memories, unsupported = discover_memories(sources)
    reachable = reachable_modules(sources, top_module)
    if reachable is not None:
        memories = [memory for memory in memories if memory.name in reachable]
        unsupported = [row for row in unsupported if row["module"] in reachable]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    wrappers: list[str] = []
    manifest: list[dict[str, object]] = unsupported[:]
    replaced_sources: set[Path] = set()
    used_macros: list[Macro] = []
    for memory in memories:
        try:
            macro = choose_macro(memory, macros)
            wrapper, row = emit_wrapper(memory, macro)
        except Exception as exc:
            manifest.append(
                {"module": memory.name, "source": str(memory.source), "status": "unsupported", "reason": str(exc)}
            )
            continue
        wrappers.append(wrapper)
        manifest.append(row)
        replaced_sources.add(memory.source.resolve())
        used_macros.append(macro)

    wrapper_path = output / "sram22_replacements.sv"
    blackbox_path = output / "sram22_blackboxes.v"
    manifest_path = output / "sram22_manifest.json"
    wrapper_path.write_text("\n".join(wrappers))
    blackbox_path.write_text(emit_blackboxes(used_macros))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    macro_liberties = sorted({str(macro.liberty) for macro in used_macros})
    macro_area = sum(float(row.get("macro_total_area", 0.0)) for row in manifest)
    remaining_sources = [path for path in source_paths if Path(path).resolve() not in replaced_sources]
    extra_sources = [str(blackbox_path), str(wrapper_path)] if used_macros else []
    return {
        "source_paths": extra_sources + remaining_sources,
        "macro_liberties": macro_liberties,
        "macro_area": macro_area,
        "mapped": len(used_macros),
        "manifest": str(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-list", type=Path, help="file containing one Verilog source path per line")
    parser.add_argument("--source-dir", type=Path, help="directory to scan when --source-list is omitted")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sky130-root", default=os.environ.get("SKY130_ROOT"))
    parser.add_argument("--top", default=None)
    args = parser.parse_args()

    if args.source_list:
        sources = [line.strip() for line in args.source_list.read_text().splitlines() if line.strip()]
    elif args.source_dir:
        sources = [str(path) for path in sorted(args.source_dir.glob("*.sv"))]
    else:
        parser.error("one of --source-list or --source-dir is required")

    result = prepare_ip_replacement(sources, str(args.output_dir), args.sky130_root, args.top)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
