#!/usr/bin/env python3
"""ELF -> hex converter for Buckyball P2E DDR loading.

Output format:
    @0
    XX
    XX
    ...
where XX is one byte per line (uppercase hex).

The hex file represents bytes starting at DDR offset 0, which maps to
CPU address 0x80000000. All LOAD segments are concatenated by physical
address, with gaps padded by zeros.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def elf_to_bin(elf_path: Path, bin_path: Path, objcopy: str) -> None:
    subprocess.run(
        [objcopy, "-O", "binary", str(elf_path), str(bin_path)],
        check=True,
    )


def bin_to_hex(bin_path: Path, hex_path: Path) -> int:
    data = bin_path.read_bytes()
    lines = ["@0"]
    lines.extend(f"{b:02X}" for b in data)
    hex_path.write_text("\n".join(lines) + "\n")
    return len(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("elf", type=Path, help="Input ELF file")
    parser.add_argument(
        "hex",
        type=Path,
        nargs="?",
        help="Output hex file (default: <elf>.hex)",
    )
    parser.add_argument(
        "--objcopy",
        default="riscv64-unknown-elf-objcopy",
        help="objcopy binary (default: riscv64-unknown-elf-objcopy)",
    )
    args = parser.parse_args()

    if not args.elf.is_file():
        print(f"error: ELF not found: {args.elf}", file=sys.stderr)
        return 1

    hex_path = args.hex or args.elf.with_suffix(".hex")
    bin_path = hex_path.with_suffix(".bin")

    try:
        elf_to_bin(args.elf, bin_path, args.objcopy)
    except FileNotFoundError:
        print(f"error: objcopy not found: {args.objcopy}", file=sys.stderr)
        print("hint: run inside `nix develop` shell", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"error: objcopy failed: {e}", file=sys.stderr)
        return 1

    n = bin_to_hex(bin_path, hex_path)
    bin_path.unlink()

    print(f"wrote {hex_path} ({n} bytes, {n + 1} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
