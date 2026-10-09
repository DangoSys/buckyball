# Validation record

Verified on 2026-10-09 for **Arty A7-100T**, part `xc7a100tcsg324-1`.
These results cover RTL simulation and FPGA implementation. Physical JTAG
programming, USB UART operation and board acceptance remain untested.

## Reproducibility

The build exports 264 unchanged Buckyball source files from commit
`1f3d54535bd5fb76d5fd2f6b05ed36e591cf3097`, checking each SHA-256 against
`UPSTREAM_MANIFEST.json`. All five dependencies were exported from the commits
in `dependencies.json`. The verified build used local Git objects for the
Buckyball baseline and freshly downloaded dependency snapshots.

Tools: Vivado 2024.1 (implementation), Vivado XSim 2025.1 (interface tests),
Verilator 5.050, RISC-V GCC 13.2.0, Java 17, sbt 1.12.11, Scala 2.13.16,
Chisel 6.7.0 and firtool 1.62.1. The firmware compiler is selected with
`RISCV_PREFIX`; different compiler versions can produce different firmware.

| Artifact | SHA-256 |
|---|---|
| Original Chisel output, `PebbleA7Chip.emitted.txt` | `54af3762e65b35e39367f8cbecc96298f11da54d923d54f79b739867470c65db` |
| Packaged RTL, `PebbleA7Chip.sv` | `51908c47eafa5dbf0cfe1004c86dee50331be081e6d716ae0f387a67ad3289a9` |
| Resident firmware, `demo.bin` (2440 bytes) | `5a9f143857c418d689d7e9bd0be2997ebf5a0812329aa6ffddd1c99d337d6df7` |

Removing the unused FireSim `targetutils` build dependency produced identical
RTL. No upstream CPU, DMA or operator implementation was modified.

## Functional checks

- Four XSim suites passed: AXI BRAM, AXI UART, UART receiver and serial loader.
  They exercise bursts, byte strobes, backpressure, malformed requests,
  FIFO handling, CRC/range errors and reset recovery.
- Four Python host protocol tests passed, including fragmented replies,
  sequence wrap, a known CRC vector, and invalid response rejection.
- Full-system Verilator simulation passed in **1,724,701 clock cycles**:
  cold resident boot before any upload; serial firmware upload and readback;
  CPU RAM/MMIO checks; DMA and original custom instructions for SMatMul,
  MatAdd and ReLU; UART echo; a second input seed; and reset/reboot.
  Assertions were enabled. Only DPI/diagnostic printing was disabled.
  The simulation uses 16 clocks per UART bit to reduce runtime; this cycle
  count is a test duration, not an accelerator performance benchmark.
- Black, Flake8, LLVM clang-format, Scalafmt, Verible, EOF, whitespace,
  conflict-marker and file-size checks passed via the installed hook tools.
  The pre-commit wrapper could not finish fetching hook repositories through
  the server's proxy; this is not recorded as a successful pre-commit run.

## Routed implementation

The complete top-level design passed synthesis, placement, routing and
`write_bitstream`. No DSP-forcing attributes or alternative mapping experiment
were used. DSP usage below is for the whole chip, not a count of SMatMul PEs.

| Metric | Measured result |
|---|---:|
| Core clock | 50 MHz (20 ns), from the 100 MHz board oscillator |
| Slice LUTs | 43,405 / 63,400 (68.46%) |
| Slice registers | 24,253 / 126,800 (19.13%) |
| Block RAM | 65.5 / 135 tiles (63 RAMB36 + 5 RAMB18) |
| DSP48E1 | 12 / 240 |
| Setup WNS / TNS | +1.004 ns / 0 ns |
| Hold WHS / THS | +0.018 ns / 0 ns |
| Pulse-width slack | +3.000 ns |
| DRC | 0 errors, 17 DSP pipeline performance warnings |
| CDC | 3 safe crossings; 0 unsafe/unknown crossings |

`audit_routed.tcl` passed: correct part, all nine package pins and LVCMOS33
standards, 50 MHz generated clock, and no latches or blackboxes. Timing
exceptions cover asynchronous input synchronizers, MMCM-lock reset assertion,
and asynchronous LED/UART outputs; they do not validate external board wiring.

The implementation scripts retain timing, utilization, routing, DRC, CDC and
I/O reports under `build/arty/`. Generated binaries and checkpoints are excluded
from Git. The recorded tests are evidence for this configuration, not exhaustive
ISA, protocol, workload-size or silicon validation.
