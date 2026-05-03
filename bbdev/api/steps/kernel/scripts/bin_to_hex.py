#!/usr/bin/env python3

import subprocess
import sys
import os

def bin_to_hex(bin_file, hex_file, objcopy_path="riscv64-unknown-elf-objcopy"):
    """
    将 RISC-V bin 文件转换为 Verilog readmemh 格式
    地址格式：@0xNNNNNNNN
    """
    print(f"Converting {bin_file} -> {hex_file}")

    # Step 1: bin -> Intel HEX
    temp_hex = f"{bin_file}.tmp.hex"
    try:
        cmd = [objcopy_path, "-O", "ihex", bin_file, temp_hex]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: objcopy failed: {result.stderr}")
            return False

        # Step 2: Parse Intel HEX and convert to readmemh format
        memory_data = {}
        current_base_address = 0

        with open(temp_hex, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith(':'):
                    continue

                record = line[1:]
                if len(record) < 8:
                    continue

                byte_count = int(record[0:2], 16)
                address = int(record[2:6], 16)
                record_type = int(record[6:8], 16)

                if record_type == 0x00:  # Data record
                    data_start = 8
                    for i in range(byte_count):
                        if data_start + 2 <= len(record) - 2:
                            byte_data = record[data_start:data_start+2]
                            memory_address = current_base_address + address + i
                            memory_data[memory_address] = int(byte_data, 16)
                            data_start += 2

                elif record_type == 0x04:  # Extended linear address
                    if byte_count == 2 and len(record) >= 12:
                        high_address = int(record[8:12], 16)
                        current_base_address = high_address << 16

        if not memory_data:
            print("Error: No valid data found")
            return False

        # Step 3: Write readmemh format
        with open(hex_file, 'w') as f:
            sorted_addresses = sorted(memory_data.keys())
            current_address = None

            for addr in sorted_addresses:
                # New address section if not continuous
                if current_address is None or addr != current_address + 1:
                    f.write(f"@0x{addr:08X}\n")
                    current_address = addr
                else:
                    current_address = addr

                byte_value = memory_data[addr]
                f.write(f"{byte_value:02X}\n")

        print(f"Successfully generated {hex_file}")
        print(f"Address range: 0x{sorted_addresses[0]:08X} - 0x{sorted_addresses[-1]:08X}")
        print(f"Total bytes: {len(memory_data)}")
        return True

    except Exception as e:
        print(f"Error: Conversion failed - {e}")
        return False
    finally:
        if os.path.exists(temp_hex):
            os.remove(temp_hex)

def main():
    if len(sys.argv) != 3:
        print("Usage: bin_to_hex.py <input_bin> <output_hex>")
        sys.exit(1)

    bin_file = sys.argv[1]
    hex_file = sys.argv[2]

    if not os.path.exists(bin_file):
        print(f"Error: File not found: {bin_file}")
        sys.exit(1)

    success = bin_to_hex(bin_file, hex_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
