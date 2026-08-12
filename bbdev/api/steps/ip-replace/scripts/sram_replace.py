"""Prepare technology-neutral SRAM collateral for synthesis.

The elaborator has already split sequential memories into separate SystemVerilog
modules. This module preserves those implementations, selects the RTL reachable
from a synthesis top, and records memory interfaces in a manifest. A PDK-owned
flow can consume that manifest to choose macros and replace modules later.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


MODULE_NAME_RE = re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)\b", re.M)
MODULE_HEADER_RE = re.compile(
    r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)\s*(?:#\s*\(.*?\)\s*)?\((.*?)\)\s*;",
    re.S,
)
MEMORY_RE = re.compile(
    r"\breg\s+(?:\[\s*(\d+)\s*:\s*(\d+)\s*\]\s+)?Memory\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]",
    re.S,
)
RANGE_RE = re.compile(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]\s*{name}\b")
INSTANCE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s*(?:#\s*\(.*?\)\s*)?([A-Za-z_][A-Za-z0-9_$]*)\s*\(",
    re.M | re.S,
)


@dataclass(frozen=True)
class Memory:
    name: str
    source: Path
    depth: int
    width: int
    port_type: str
    address_width: int | None
    write_mask_width: int | None


def width_from_range(text: str, name: str) -> int | None:
    match = re.search(RANGE_RE.pattern.format(name=re.escape(name)), text)
    if match is None:
        return None
    return abs(int(match.group(1)) - int(match.group(2))) + 1


def ceil_log2(value: int) -> int:
    return max(1, math.ceil(math.log2(value)))


def module_sources(source_paths: list[Path]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for source in source_paths:
        match = MODULE_NAME_RE.search(source.read_text(errors="replace"))
        if match is not None:
            sources.setdefault(match.group(1), source)
    return sources


def reachable_modules(source_paths: list[Path], top_module: str) -> set[str]:
    sources = module_sources(source_paths)
    if top_module not in sources:
        raise ValueError(f"synthesis top is not defined in source list: {top_module}")

    module_text = {
        module: path.read_text(errors="replace") for module, path in sources.items()
    }
    reachable = {top_module}
    pending = [top_module]
    while pending:
        module = pending.pop()
        for child, _instance in INSTANCE_RE.findall(module_text[module]):
            if child in sources and child not in reachable:
                reachable.add(child)
                pending.append(child)
    return reachable


def classify_ports(header: str) -> str:
    names = set(re.findall(r"\b((?:RW|R|W)\d+)_[A-Za-z0-9_]+\b", header))
    rw_ports = {name for name in names if name.startswith("RW")}
    if rw_ports:
        return f"{len(rw_ports)}RW"
    reads = {name for name in names if name.startswith("R")}
    writes = {name for name in names if name.startswith("W")}
    return f"{len(reads)}R{len(writes)}W"


def discover_memories(source_paths: list[Path]) -> list[Memory]:
    memories: list[Memory] = []
    for source in source_paths:
        text = source.read_text(errors="replace")
        module_match = MODULE_HEADER_RE.search(text)
        memory_match = MEMORY_RE.search(text)
        if module_match is None or memory_match is None:
            continue
        name, header = module_match.groups()
        data_msb = int(memory_match.group(1) or 0)
        data_lsb = int(memory_match.group(2) or 0)
        depth_a = int(memory_match.group(3))
        depth_b = int(memory_match.group(4))
        depth = abs(depth_a - depth_b) + 1
        memories.append(
            Memory(
                name=name,
                source=source,
                depth=depth,
                width=abs(data_msb - data_lsb) + 1,
                port_type=classify_ports(header),
                address_width=width_from_range(header, "RW0_addr") or ceil_log2(depth),
                write_mask_width=(
                    width_from_range(header, "RW0_wmask")
                    or (1 if "RW0_wmask" in header else None)
                ),
            )
        )
    return memories


def prepare_sram_collateral(
    source_paths: list[str], output_dir: str, top_module: str
) -> dict[str, object]:
    """Emit top-scoped source and memory manifests without choosing a PDK macro."""
    sources = [Path(path).resolve() for path in source_paths]
    reachable = reachable_modules(sources, top_module)
    source_modules = module_sources(sources)
    selected_paths = {source_modules[module] for module in reachable}
    selected = [path for path in sources if path in selected_paths]
    memories = discover_memories(selected)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "sram_manifest.json"
    manifest = {
        "schema_version": 1,
        "top_module": top_module,
        "memories": [
            {
                "module": memory.name,
                "source": str(memory.source),
                "depth": memory.depth,
                "width": memory.width,
                "port_type": memory.port_type,
                "address_width": memory.address_width,
                "write_mask_width": memory.write_mask_width,
            }
            for memory in memories
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "source_paths": [str(path) for path in selected],
        "sram_manifest": str(manifest_path),
        "sram_memory_count": len(memories),
        "top_module": top_module,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-list", type=Path, help="one Verilog source path per line")
    parser.add_argument("--source-dir", type=Path, help="directory to scan when --source-list is omitted")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", required=True)
    args = parser.parse_args()

    if args.source_list:
        sources = [line.strip() for line in args.source_list.read_text().splitlines() if line.strip()]
    elif args.source_dir:
        sources = [str(path) for path in sorted(args.source_dir.glob("*.sv"))]
    else:
        parser.error("one of --source-list or --source-dir is required")

    result = prepare_sram_collateral(sources, str(args.output_dir), args.top)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
