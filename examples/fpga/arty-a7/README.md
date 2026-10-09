# Arty A7 standalone chip

An independent FPGA platform for the **Digilent Arty A7-100T**, Vivado part
`xc7a100tcsg324-1`. It integrates the original Chisel RocketBB CPU, Buckyball
frontend/DMA, SMatMul 8x16, MatAdd and ReLU with synthesizable board peripherals.
The AXI RAM, UART peripheral/loader, clock/reset wrapper, pin constraints and
bare-metal firmware are implemented here. No K7 DDR/MIG or P2E harness is used.

## Configuration

- One RV64IMAC core, 4 KiB instruction and data caches; no VM, FPU or debug module.
- Eight 512x128-bit scratchpad banks (64 KiB total).
- 128 KiB BRAM program/data memory at `0x80000000`, boot ROM at `0x10000`.
- AXI UART/status at `0x60000000`; 16-byte receive FIFO, 115200 baud, 8N1.
- Board oscillator: 100 MHz; MMCM core clock: 50 MHz.

This example deliberately pins Buckyball commit
`1f3d54535bd5fb76d5fd2f6b05ed36e591cf3097` and the dependency commits in
`dependencies.json`, rather than changing the framework's default chip or build.
The recorded implementation results apply to that baseline. Migrating to a newer
framework revision requires regeneration and verification.

## Build

Run all commands from this directory. Linux prerequisites: Python 3.10+, Git,
curl, Java, sbt, RV64 bare-metal GCC/binutils, Verilator, and Vivado with Artix-7
device support. Tested versions are in [VALIDATION.md](VALIDATION.md).

```bash
# Fetch exact dependency snapshots, compile firmware, elaborate Chisel and wire AXI.
bash scripts/elaborate.sh
python3 tests/test_host.py
bash tests/simulate_chip.sh
vivado -mode batch -source board/build_arty.tcl
vivado -mode batch -source board/audit_routed.tcl
```

`prepare_sources.py` exports original Git blobs and checks all 264 hashes in
`UPSTREAM_MANIFEST.json`. It never overwrites changed source files. Downloads
honor curl's standard proxy environment. Dependency licenses remain alongside
the exported sources. Chisel 6.7.0, Scala 2.13.16, sbt 1.12.11 and firtool 1.62.1
are the validated toolchain; the separate sbt build does not override the main
repository's toolchain.

`prepare_rtl.py` separates firtool's trailing ancillary `.f` list from its
concatenated output without editing RTL logic. Original emitted bytes and both
hashes remain under `generated/`. `make_platform.py` regenerates explicit AXI
wiring when the Chisel top changes.

Windows interface tests require Vivado on PATH, or an explicit path:

```powershell
.\tests\simulate_units.ps1 -VivadoBin "C:\Xilinx\2025.1\Vivado\bin"
```

Generated RTL, dependencies, firmware binaries, checkpoints and bitstreams are
build products and are not committed. The bitstream is `build/arty/pebble_a7.bit`.
The build fails on setup/hold violations or DRC errors. The additional routed
audit verifies the device, all nine pins, clock, absence of latches and blackboxes.

## Board operation

1. Connect the Arty PROG/UART USB port. Load the bitstream using Vivado Hardware
   Manager or `program.ps1`. The script requires exactly one XC7A100T target and
   configures volatile FPGA SRAM; it does not program flash.
2. Open a 115200 8N1 terminal with flow control disabled. Set SW0 high and press
   BTN0 to run the resident firmware. LED0 indicates software success, LED1 a
   loader/software error, LED2 CPU released from reset, and LED3 heartbeat.
3. Expect `PASS CPU_RAM_MMIO`, `PASS SMATMUL_8x16x16`, `PASS MATADD_I32`,
   `PASS RELU_I32`, and `PASS PEBBLE_A7_FULL_CHIP`. Send `r` to rerun with a new
   input pattern; other bytes echo. LED2 alone does not indicate successful work.

To upload a program, set SW0 low, press BTN0 and close other serial terminals:

```bash
python3 -m pip install pyserial
python3 host/load.py --port /dev/ttyUSB1 firmware/demo.bin
# Windows: use --port COM5 (replace with the board's actual port).
```

The host writes and reads back every block before RUN. Images must be linked at
`0x80000000`. BTN0 preserves RAM: SW0 high boots the **current RAM contents**.
Reconfigure the FPGA to restore the original resident image after uploading a
different program. Reset is required to return from running software to the loader.

## Interfaces and verification

[PROTOCOL.md](PROTOCOL.md) documents the loader packets and MMIO registers.
`tests/` covers burst requests, byte enables, backpressure, CRC/range errors and
reset recovery. Full-system Verilator simulation drives the serial pins to load
actual RISC-V firmware, which executes the original custom instructions and
checks DMA results against C references for all three operators. Cold boot is tested
before any upload, so the loader cannot mask a missing BRAM initialization image.

See [VALIDATION.md](VALIDATION.md) for measured resources and routed timing.
**Physical board programming and serial acceptance have not been performed.**
This is a bare-metal BRAM platform; DDR, Ethernet and Linux are outside its scope.
