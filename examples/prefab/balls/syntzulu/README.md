# SYNtzulu on Buckyball Toy

The opt-in `ToySyntzuluConfig` runs the upstream dual-lane SNN datapath under Rocket control. This prototype has four dense layers of 16 neurons each, 16 binary input channels and 16-bit membrane potentials. The workload supplies already encoded spikes. Thresholds are 1 for the three hidden layers and 1024 for the output layer. The core keeps membrane state across input frames.

## RoCC interface

Use one allocated result bank per producer. Instructions have one iteration and write a zero-extended result to its first row. The adapter waits for the SRAM acknowledgement before retiring the ROB entry.

| Command | funct7 in this configuration | rs2 |
| --- | ---: | --- |
| SYNTZULU_WEIGHT | 40 | Bit 39 selects the pair of weight memories, bits 38–32 select a word, bits 31–0 hold two 16-bit slices |
| SYNTZULU_RUN | 41 | Bits 15–0 hold one frame, consumed in four groups starting with the low nibble. Completion waits for the final layer |
| SYNTZULU_READ | 42 | Bits 2–0 select one of eight output words. The low and high halves are the two lanes' membrane potentials |

These numbers are scoped to the selected Toy configuration. The C driver in `workloads/isa/syntzulu.h` obtains them from the generated manifest. The current configuration instantiates only SYNtzulu. BEMU support is outside this RTL prototype.

## Stateful four-frame regression

After reset, the test programs all 128 addresses of the four weight memories once, then submits frames `0001`, `0000`, `8000` and `ffff` without resetting the core. The first frame produces `0x00280068` in output pair zero. On the silent frame, its two retained membrane potentials decay from 40 and 104 to 35 and 93. Later frames drive some outputs across threshold 1024 and exercise reset-on-spike. The C program calculates each expected potential independently using integer decay and checks all 32 output words. The standalone RTL test checks the same sequence against recorded upstream netlist outputs. The weights are synthetic.

From the repository root in the Buckyball development environment:

```sh
bbdev workload --build '--config toy/devices/toy/configs/ToySyntzuluConfig.toml --device toy --ctest'
bbdev verilator --run '--config toy/devices/toy/configs/ToySyntzuluConfig.toml --device toy --test toy-hart0-ctest-syntzulu_frame-baremetal --batch'
```

To run the direct core test without Rocket:

```sh
ball=examples/prefab/balls/syntzulu
mkdir -p "$ball/output"
verilator --binary --timing -Wno-fatal --top-module reference_tb \
  --Mdir "$ball/output/reference" arch/src/main/resources/syntzulu/SyntzuluCore.sv "$ball/verify/ReferenceCore.sv" "$ball/verify/reference_tb.sv"
"$ball/output/reference/Vreference_tb"
```

The same fixture passed against the unmodified upstream core after sv2v conversion and Yosys process lowering. To run that independent reference with a local upstream checkout, run `verify/check-upstream.sh /path/to/SYNtzulu` from this component. The Toy RTL test used Verilator 5.046, the BBSim DRAM backend, DRAMSim3, and a lightweight host monitor on macOS. Hart 0 exited successfully at simulated cycle 17577, including boot, weight programming and readback.

## RTL provenance

The bundled source combines the required modules from [gianlucaleone/SYNtzulu at c9de580](https://github.com/gianlucaleone/SYNtzulu/tree/c9de5808376222c2de17e4f5771b4996b8b507fd/rtl/syntzulu). The compilation unit selects the upstream inferred BRAM implementation. Two clocked assignments use nonblocking updates in this port: the layer counter increment in `snn_lp` and the reset of `done` in `stack`. The original mixed blocking/nonblocking assignments made RTL simulation sensitive to scheduling across modules. The upstream license and third-party inventory accompany the RTL in `arch/src/main/resources/syntzulu`.
