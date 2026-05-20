#!/usr/bin/env python3

import sys
import os


def bin_to_hex(bin_file, hex_file, base_address=0):
    """
    Convert a raw binary file to Verilog readmemh format.

    The output uses one byte per line with `@0xADDR` markers for the starting
    offset (relative to base_address). The input is treated as a contiguous
    byte stream starting at base_address; the resulting `@` offset is always 0
    because raw binaries have no internal address information.

    base_address is kept in the signature for API compatibility and is only
    used to print the absolute address range.
    """
    print(f"Converting {bin_file} -> {hex_file}")
    if base_address != 0:
        print(f"Base address offset: 0x{base_address:08X}")

    try:
        with open(bin_file, "rb") as f:
            data = f.read()

        if not data:
            print("Error: Input file is empty")
            return False

        with open(hex_file, "w") as f:
            f.write("@0\n")
            for byte in data:
                f.write(f"{byte:02X}\n")

        print(f"Successfully generated {hex_file}")
        print(
            f"Address range: 0x{base_address:08X} - 0x{base_address + len(data) - 1:08X}"
        )
        print(f"Total bytes: {len(data)}")
        return True

    except Exception as e:
        print(f"Error: Conversion failed - {e}")
        return False


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: bin_to_hex.py <input_bin> <output_hex> [base_address]")
        print("  base_address: optional hex address for reporting (e.g., 0x80000000)")
        sys.exit(1)

    bin_file = sys.argv[1]
    hex_file = sys.argv[2]
    base_address = 0

    if len(sys.argv) == 4:
        base_str = sys.argv[3]
        if base_str.startswith("0x") or base_str.startswith("0X"):
            base_address = int(base_str, 16)
        else:
            base_address = int(base_str, 10)

    if not os.path.exists(bin_file):
        print(f"Error: File not found: {bin_file}")
        sys.exit(1)

    success = bin_to_hex(bin_file, hex_file, base_address=base_address)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
