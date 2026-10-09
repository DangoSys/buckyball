#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cc="${RISCV_PREFIX:-riscv64-unknown-elf-}gcc"
objcopy="${RISCV_PREFIX:-riscv64-unknown-elf-}objcopy"
"$cc" -march=rv64imac_zicsr_zifencei -mabi=lp64 -mcmodel=medany -mno-relax \
  -msmall-data-limit=0 -O2 -ffreestanding -fno-builtin -nostdlib -Wall -Wextra \
  -Wl,--no-relax,--build-id=none,-T,firmware/link.ld,-Map,firmware/demo.map \
  firmware/start.S firmware/main.c -o firmware/demo.elf
"$objcopy" -O binary firmware/demo.elf firmware/demo.bin
"$cc" -march=rv64imac -mabi=lp64 -nostdlib -Wl,-Ttext=0x10000,--build-id=none \
  firmware/bootrom.S -o firmware/bootrom.elf
"$objcopy" -O binary firmware/bootrom.elf firmware/bootrom.bin
python3 - <<'PY'
from pathlib import Path
data = Path('firmware/demo.bin').read_bytes()
assert len(data) < 128*1024-4096
padded = data + bytes((-len(data)) % 16)
Path('firmware/demo.hex').write_text(''.join(padded[i:i+16][::-1].hex()+'\n' for i in range(0,len(padded),16)))
print(f'Firmware: {len(data)} bytes')
PY
