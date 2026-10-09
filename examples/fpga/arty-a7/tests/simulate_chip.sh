#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build/chip-sim
verilator --cc --exe --build -j 8 --assert -Wno-fatal --top-module pebble_a7_platform \
  -DBUCKYBALL_DISABLE_DPI -DPRINTF_COND=0 -GUART_DIVISOR=16 --Mdir build/chip-sim \
  -CFLAGS '-std=c++17 -O2' generated/PebbleA7Chip.sv \
  board/pebble_a7_platform.sv board/axi_bram.sv board/axi_uart.sv \
  board/uart_loader.sv board/uart_rx.sv board/uart_tx.sv "$(pwd)/tests/chip_sim.cpp" \
  > build/chip-sim/compile.log 2>&1
build/chip-sim/Vpebble_a7_platform | tee build/chip-sim/run.log
