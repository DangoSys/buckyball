#!/usr/bin/env python3
"""ELF and optional DDR image -> hex converter for Buckyball P2E loading.

Output format:
    @0x0
    XX
    XX
    ...
where XX is one byte per line (uppercase hex).

Address markers are DDR offsets relative to CPU address 0x80000000.
Preserve separate ELF sections without materializing address gaps.
"""

import argparse
import binascii
import subprocess
import sys
from pathlib import Path


def elf_to_hex(
    elf_path: Path, hex_path: Path, objcopy: str, ddr_image: Path | None = None
) -> int:
    count = 0
    staged = hex_path.with_suffix(hex_path.suffix + ".tmp")
    with staged.open("wb") as output:
        # Match the simulator: initialize the complete DDR image, then overlay the ELF.
        if ddr_image is not None:
            output.write(b"@0x0\n")
            with ddr_image.open("rb") as image:
                while data := image.read(1024 * 1024):
                    output.write(binascii.hexlify(data, b"\n", 1).upper() + b"\n")
                    count += len(data)
        with subprocess.Popen(
            [objcopy, "-O", "verilog", str(elf_path), "/dev/stdout"],
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            for line in process.stdout:
                if line.startswith("@"):
                    offset = int(line[1:], 16) - 0x80000000
                    if offset < 0:
                        raise ValueError(f"ELF section is below DDR base: {line}")
                    output.write(f"@0x{offset:X}\n".encode())
                else:
                    data = bytes.fromhex(line)
                    if data:
                        output.write(binascii.hexlify(data, b"\n", 1).upper() + b"\n")
                        count += len(data)
            returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, process.args)
    staged.replace(hex_path)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("elf", type=Path, help="Input ELF file")
    parser.add_argument(
        "--ddr-image",
        type=Path,
        help="DDR image loaded at DDR offset zero before the ELF",
    )
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
    try:
        n = elf_to_hex(args.elf, hex_path, args.objcopy, args.ddr_image)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        print("hint: run inside `nix develop` shell", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"error: objcopy failed: {e}", file=sys.stderr)
        return 1

    print(f"wrote {hex_path} ({n} payload bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
