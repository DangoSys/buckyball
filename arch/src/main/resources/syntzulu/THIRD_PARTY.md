# Third-Party Components

This document identifies third-party source code included in SYNtzulu, including locally modified copies. The project-level MIT license does not replace the licenses applicable to third-party portions. Copyright and license notices for those portions remain applicable.

Source inventory reviewed against SYNtzulu commit `c6ac00f1103d6996751acde43f82edcc8b943f4b`. Provenance information updated on 2026-10-29 following confirmation by the project maintainer.

## 1. SERV RISC-V core

- **Upstream:** https://github.com/olofk/serv
- **License:** ISC, except for the compressed-instruction decoder listed separately below.
- **License reference:** https://github.com/olofk/serv/blob/1.2.1/LICENSE
- **Files:**

- `rtl/serv/serv_aligner.v`
- `rtl/serv/serv_alu.v`
- `rtl/serv/serv_bufreg.v`
- `rtl/serv/serv_bufreg2.v`
- `rtl/serv/serv_csr.v`
- `rtl/serv/serv_ctrl.v`
- `rtl/serv/serv_decode.v`
- `rtl/serv/serv_immdec.v`
- `rtl/serv/serv_mem_if.v`
- `rtl/serv/serv_rf_if.v`
- `rtl/serv/serv_rf_ram.v`
- `rtl/serv/serv_rf_ram_if.v`
- `rtl/serv/serv_rf_top.v`
- `rtl/serv/serv_state.v`
- `rtl/serv/serv_top.v`

Most of these files match SERV 1.2.1; others contain changes or match another SERV revision. The mapping includes adapted SERV code, not a claim that every file is an unmodified copy.

## 2. SERV compressed-instruction decoder (Ibex-derived)

- **Upstream as integrated:** https://github.com/olofk/serv/blob/1.2.1/rtl/serv_compdec.v
- **Original project:** https://github.com/lowRISC/ibex
- **License:** Apache-2.0, explicitly declared in the local file header.
- **Attribution:** lowRISC contributors; ETH Zurich and University of Bologna. The header credits the adaptation to SERV to @Abdulwadoodd.
- **File:** `rtl/serv/serv_compdec.v`

This file is an exception to the ISC license used for the rest of the SERV core.

## 3. Servant reference platform and simulation support

- **Upstream:** https://github.com/olofk/serv
- **License:** ISC for upstream-derived portions.
- **Reference platform:** https://github.com/olofk/serv/tree/1.2.1/servant
- **Reference testbenches:** https://github.com/olofk/serv/tree/1.2.1/bench
- **Files containing copied or adapted platform/testbench code:**

- `rtl/servant/new_servant.v`
- `rtl/servant/new_service.v`
- `rtl/servant/servant_arbiter.v`
- `rtl/servant/servant_gpio.v`
- `rtl/servant/servant_mux.v`
- `rtl/servant/servant_ram.v`
- `rtl/servant/servant_timer.v`
- `rtl/servant/servant_slow_timer.v`
- `sim/tb/servant_sim.v`
- `sim/tb/servant_tb.v`
- `sim/tb/uart_decoder.v`

`new_servant.v` adapts upstream `servant.v`. `new_service.v` derives from upstream `service.v`, with SYNtzulu-specific additions. 
`servant_slow_timer.v` adapts the Servant timer structure. The testbenches contain SYNtzulu-specific additions. `servant_arbiter.v` and `servant_ram.v` match SERV 1.2.1; `uart_decoder.v` matches the upstream decoder.

This list does not assign the SERV license to the entire `rtl/servant/` directory: that directory also contains SYNtzulu-specific logic and components with other licenses.

## 4. SERV binary-to-hexadecimal conversion utility

- **Upstream:** https://github.com/olofk/serv/blob/main/sw/makehex.py
- **License/status:** Public-domain dedication stated in the file header (Unlicense-style notice).
- **File:** `firmware/makehex.py`

The local file matches the SERV utility. Its explicit public-domain notice is recorded separately from SERV's project-level ISC license; the local file does not contain a complete Unlicense text or an SPDX identifier.

## 5. BasicUART serial transmitter

- **Upstream:** https://github.com/STjurny/BasicUART
- **Source:** https://github.com/STjurny/BasicUART/blob/master/SerialTransmitter.v
- **License:** MIT, explicitly stated in the source header.
- **Copyright:** 2020 Stanislav Jurny.
- **File:** `rtl/servant/SerialTransmitter.v`

The local transmitter is an adapted copy of the upstream module.

## 6. ice40_power glitch-free clock multiplexer

- **Upstream:** https://github.com/tinyvision-ai-inc/ice40_power
- **Source:** https://github.com/tinyvision-ai-inc/ice40_power/blob/master/lp_design/gfcm.sv
- **License:** MIT.
- **License reference:** https://github.com/tinyvision-ai-inc/ice40_power/blob/master/LICENSE
- **Copyright in upstream license:** 2020 vr2045.
- **File:** `rtl/servant/gfcm.v`

The local file matches upstream `lp_design/gfcm.sv`, with a different filename extension. Its header also cites https://vlsitutorials.com/glitch-free-clock-mux/ as the design reference.

## 7. PicoRV32 / PicoSoC SPI flash simulation model

- **Upstream:** https://github.com/YosysHQ/picorv32
- **Source:** https://github.com/YosysHQ/picorv32/blob/main/picosoc/spiflash.v
- **License:** ISC, with the license notice retained in the local file.
- **Copyright:** 2017 Claire Xenia Wolf.
- **File:** `sim/tb/flash_spi_sim.sv`

This is an adapted copy of the PicoSoC flash model, with changes to initialization, memory layout and address handling. This entry identifies the flash simulation model; it does not imply that a PicoRV32 CPU core is included in SYNtzulu.

## 8. Yosys iCE40 cell simulation library

- **Upstream:** https://github.com/YosysHQ/yosys
- **Reference source:** https://github.com/YosysHQ/yosys/blob/yosys-0.30/techlibs/ice40/cells_sim.v
- **License:** ISC.
- **License reference:** https://github.com/YosysHQ/yosys/blob/yosys-0.30/COPYING
- **Files:**

- `sim/tb/cells_sim.v`
- `sim/tb/SB_PLL40_PAD.v`
- `sim/tb/SB_PLL40_2F_PAD.v`
- `rtl/syntzulu/SB_SPRAM256KA.v`

`cells_sim.v` closely matches the Yosys 0.30 library. The other files contain extracted/adapted module definitions from that library; the PLL files add local simulation behavior. The library also retains references to Project IceStorm timing data: https://github.com/YosysHQ/icestorm.

## 9. Verilog testbench utilities

- **Upstream:** https://github.com/fusesoc/vlog_tb_utils
- **Source:** https://github.com/fusesoc/vlog_tb_utils/blob/v1.1/vlog_tb_utils.v
- **License:** ISC, with the full notice retained in the local file.
- **Copyright:** 2016 Olof Kindgren.
- **File:** `sim/tb/vlog_tb_utils.v`

The local file matches the upstream v1.1 utility.

## 10. Five EmbedDev bare-metal RISC-V support

- **Upstream:** https://github.com/five-embeddev/riscv-scratchpad/tree/master/baremetal-startup-c/src
- **Project website:** https://five-embeddev.com/
- **License:** Unlicense for upstream-derived portions. Local SPDX identifiers are present in the support files listed below; the attribution of `main.c` is based on maintainer-confirmed derivation from the Five EmbedDev example.
- **Files:**

- `firmware/src/main.c`
- `firmware/src/startup.c`
- `firmware/src/timer.c`
- `firmware/src/timer.h`
- `firmware/src/riscv-csr.h`
- `firmware/src/riscv-interrupts.h`

These include local adaptations of the main program, startup code, timer driver and CSR definitions. `main.c` combines Five EmbedDev-derived code with SYNtzulu-specific application logic. `riscv-interrupts.h` matches the upstream file.

## 11. SiFive / Freedom Metal linker scripts

- **Original project:** https://github.com/sifive/freedom-metal
- **Matching source family:** https://github.com/five-embeddev/riscv-scratchpad/blob/master/baremetal-startup-c/src/linker.lds
- **License:** Apache-2.0 for upstream-derived portions, explicitly declared in the local `linker.lds` header. The derivation of `linkerscript.lds` from SiFive was confirmed by the project maintainer.
- **Copyright:** 2020 SiFive Inc.
- **Files:**

- `firmware/src/linker.lds`
- `firmware/src/linkerscript.lds`

`linker.lds` adapts the SiFive linker script distributed with the Five EmbedDev example. `linkerscript.lds` is also derived from SiFive code, as confirmed by the project maintainer. Memory regions and sections have been customized.

## Original SYNtzulu clock-generation module

`rtl/servant/servant_clock_gen.v` is an original SYNtzulu implementation developed by the project team, as confirmed by the project maintainer. It is covered by the project-level MIT license and is not classified as a third-party SERV or ice40_power component.

## External dependencies

The demo notebooks and Python exports import external packages, including Lava-DL, PyTorch, torchvision, NumPy, SciPy and Matplotlib. The build uses external EDA tools and a RISC-V compiler. Importing or invoking those tools does not mean their source code is included in this repository; they are not classified here as vendored components.

The demo also refers to the externally supplied Ninapro DB5 dataset. Its data-use terms are separate from the source-code licenses in this inventory.

## Notice preservation

Retain the original third-party copyright, license and attribution notices. This inventory is an index, not a replacement for full license texts and any applicable upstream notices. Several locally copied files rely on upstream project-level licenses rather than complete notices in their local headers; the corresponding texts should accompany redistribution.
