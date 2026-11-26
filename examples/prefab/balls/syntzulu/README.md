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

## Bank-to-bank SNN pipeline

`ToySnnPipelineConfig` adds a SpikeIoBall and enables `SYNTZULU_RUN_BANK` (funct7 53). One instruction reads a packed 16-bit frame from source row zero, advances SYNtzulu by one time step and stores eight signed output pairs in destination rows 0–7. Each 128-bit row holds one pair in its low 32 bits. The Ball reports completion after the eighth SRAM write acknowledgement.

The cooperating workload allocates four distinct banks. Rocket issues the commands and retrieves the final index.

| Producer | Operation | Consumer |
| --- | --- | --- |
| DMA | Store 16 signed int8 features in one 128-bit row | SpikeIoBall |
| SpikeIoBall, funct7 54 | Compare each feature with a signed threshold and pack the 16 strict-greater-than results into one word | SyntzuluBall |
| SyntzuluBall, funct7 53 | Process one frame using persistent membrane state and write 16 signed potentials | SpikeIoBall |
| SpikeIoBall, funct7 55 | Select the lowest channel index attaining the maximum | Rocket, via one row DMA readback |

The workload measures four frames from input in main memory to a class index available to Rocket. The reference uses the same Toy RTL and SYNtzulu core. Rocket encodes the frames, submits each using the original immediate command, reads eight output pairs individually and computes argmax. A second host baseline uses the same bank-input instruction as the pipeline. Rocket packs the input into memory, transfers it to the SNN, bulk-reads all eight output rows with one DMA command, and selects the winner on the CPU. Each workload starts from reset with the same weights. Cycle counts sum per-frame `rdcycle` intervals, including input handling, commands, fences and readback. They exclude boot, initial weight programming, UART and correctness checks.

| RV64 bare-metal workload | Four-frame cycles in local Verilator + DRAMSim3 run |
| --- | ---: |
| snn_host | 7755 |
| snn_bulk_host | 3918 |
| snn_pipeline | 1937 |

These measurements cover the provided synthetic data and driver implementations. A direct Rocket coprocessor, silicon power, and trained-model accuracy were outside this experiment. The local run used a lightweight monitor on macOS. To reproduce in the project devshell:

```sh
cfg=toy/devices/toy/configs/ToySnnPipelineConfig.toml
bbdev workload --build "--config $cfg --device toy --ctest"
for test in snn_host snn_bulk_host snn_pipeline snn_pipeline_check; do
  bbdev verilator --run "--config $cfg --device toy --test toy-hart0-ctest-${test}-baremetal --batch"
done
```

The check workload compares 32 output words covering all four frames with an integer neuron model. It also checks the packed input masks and tests signed argmax with tied negative scores. The three timed workloads each check the four winning indices.

## Single-Ball shared weights and input delay

The single-Ball configuration also registers RUN_BANK. In its default form, source
row zero supplies a 16-bit frame. Bits 1:0 of rs2 select a delay of 0–3 accepted
frames for that input. Advance time with zero frames to drain pending events.
Inputs arriving at the same step combine by bitwise OR. READ selector 8 writes
the 16-bit output spike mask of the most recently completed frame to row zero.

With rs2 bit 2 set, the input frame is immediate in bits 31:16 and the source
bank supplies weights. Allocate a bank with at least 64 rows of 128 bits.
Row j contains weight words 2j and 2j+1, low word first. Each 64-bit word
contains the four 16-bit slices in programming order. The adapter caches one
row, invalidates it on each command, and pauses the imported core on a miss.
This experiment retains the original BRAM path for comparison. Clock gating
in the simulation prototype requires target-specific mapping for implementation.


Additional regression covers both weight paths with frame delays 0, 1, 2, 3.
All eight whole-SoC cases exit zero. A differential sequence with seed 0x82519437,
64 input frames, changing per-frame delays, collisions combined by OR, and four
drain frames matches 1088 signed voltages and 68 spike masks. This sequence uses
weight bytes in 0..3. The randomized workload emits TRACE records for comparing
two reset-to-reset runs; its exit status alone does not assert the comparison.

The full-range sequence initially timed out on the eighth frame (0xe75b).
Final-layer output groups were being appended to the input stack. Their stale
entries combined with the next input and exceeded the stack capacity. Feedback
writes now exclude the final layer. Both whole-SoC weight paths complete the
68-step full-range test and match all 1088 potentials and 68 spike masks.
An independent integer-model regression is described below.

Reset previously restarted the state FIFO pointers while preserving valid old
values in RAM. Per-entry validity now masks old current and voltage data until
rewritten. The bridge reset test queues a delayed event, resets, then checks
four zero-input frames for zero masks and zero potentials. Weight RAM contents
are intentionally retained across this state reset.

The design boundary retains the four-layer inference controller, sparse stack,
synaptic computation and LIF. Software owns application encoding and decoding.
Input delay applies to an entire frame in accepted-frame units; independent
per-axon delay remains outside the implemented interface.


## Independent integer reference

The verify/math_model.py checker computes the four-layer network directly from
signed int8 weights, without RTL execution or captured RTL intermediates as
model inputs. It decodes the programmed weight layout and spike-channel
permutation, applies Q12 decay with truncation toward zero, signed 16-bit
wrapping, threshold comparison and reset to zero. Hidden thresholds are 1 and
the output threshold is 1024. Decay factors are 3681, 3672, 3681, 3672.
State persists between accepted frames. Software independently schedules
inputs delayed by 0–3 steps and combines coincident binary events with OR.

The whole-SoC workload uses RANDOM_WEIGHT_MASK=UINT64_MAX, USE_SHARED_WEIGHTS=0/1
and RANDOM_SEED=0x82519437, 1, 0xdeadbeef. Each run has 64 input frames and four
drain frames. All 6528 output potentials and 408 spike masks match across six
runs. For the original fault seed, a standalone RTL run additionally matches
4352 potentials across all four layers.

Run python3 verify/math_model.py --logs /path/to/logs. Expected logs are
final-random-0/1.log for the original seed, math-1-0/1.log,
math-3735928559-0/1.log, and math-layers.log. The checker verifies complete
record counts, successful SoC exit, and every compared value, then writes
math-validation.json. These randomized regressions cover three seeds.


## Matched input performance comparison

Both weight modes can use syntzulu_run_frame(weight_bank, output_bank, frame,
delay_steps, shared_weights). RUN_BANK rs2 bit 3 selects an immediate frame
in bits 31:16 with eight-row automatic output; bit 2 selects shared weights.
The original bank-input mode and earlier shared immediate mode remain valid.

The matched benchmark submits the same four masks 0001, 0000, 8000, ffff through
this helper and reads eight membrane rows through one DMA per frame. Inference
timing starts before command submission and ends after the CPU-visible DMA
readback and fence. Spike-mask and mathematical checks, UART, boot, memory
allocation and initial weight-array construction are outside inference timing.
Configuration timing covers weight programming or DMA plus a destination-bank
readback/fence completion barrier. Local mode uses 256 programming commands;
shared mode transfers the same 1 KiB through DMA in 64 128-bit rows.

Three fresh-reset Toy RTL runs per mode produced identical measurements:

| Weight mode | Configuration cycles | Four-frame inference/readback | Sum |
| --- | ---: | ---: | ---: |
| Local BRAM | 7736 | 1494 | 9230 |
| Shared SRAM | 1040 | 2438 | 3478 |

Per-frame local cycles are 408, 306, 378, 402; shared cycles are 660, 450, 630,
698. Shared configuration is 7.44 times faster, inference costs 63.2 percent
more, and configuration plus four frames costs 62.3 percent fewer cycles.
This supersedes the earlier unequal-input inference comparison. These are
synthetic-workload measurements of the two current loading implementations;
local weight upload uses scalar commands while shared upload uses DMA.
The new immediate-input path in each mode also matches all 1088 final-layer
potentials and 68 output masks of the independent signed-weight model.


## Draft-PR engineering checks

The final ToySnnPipelineConfig regression passes snn_host, snn_bulk_host,
snn_pipeline and snn_pipeline_check. The check now additionally submits
SYNTZULU_READ selector 8 to store spikes, followed immediately by SpikeIoBall
SPIKE_COUNT (funct7 56). That consumer reads the producer's SRAM row and writes
its population count to a result bank. Counts are 0, 0, 3, 2 for the four frames.
The CPU observes the count; it does not read and resubmit the spike mask between
the producer and consumer. SPIKE_COUNT is optional in the cooperating config.
The single SyntzuluBall config needs no SpikeIoBall.

The original single-Ball syntzulu_frame workload also passes with all 32 words
checked. Whole-SoC reset is tested for both weight modes by compiling
syntzulu_weights.c with HOST_RESET_TEST and an external host that pulses SoC
reset for eight cycles at the first RESET_HERE marker. The marker follows
four checked frames and a command that queues an all-ones event delayed one
step. The host restarts the CPU once, then allows the same program to complete
its golden checks. This covers dirty membrane state and pending frame delay;
reset while a memory transaction is outstanding is not part of this test.

Yosys 0.61 synth_xilinx for family xc7, using sv2v, passes hierarchy and
check -assert for fixed-mode instances of the core plus bridge:

| Resource | Local weights | Shared weights |
| --- | ---: | ---: |
| LUT1..LUT6 cells | 824 | 757 |
| FDRE | 874 | 874 |
| DSP48E1 | 2 | 2 |
| RAM64M distributed memory | 48 | 0 |
| RAM32M distributed memory | 14 | 14 |
| LDCE latch | 0 | 1 |
| BUFG | 1 | 2 |

Constant shared mode removes all four local weight memories. These resources
exclude shared bank storage, the Chisel adapter and its row cache, and apply
to fixed-mode synthesis wrappers, while the runtime-dual-mode SoC retains the
local memories. Memories map to distributed RAM in this flow. This is an
inference/resource check, not place-and-route or a measured clock frequency.

The shared core still uses a low-phase latch plus a gated clock for stall.
Target implementation must map and constrain this clock path, using a
technology clock-gating cell or an FPGA clock-enable resource as appropriate.
The synthesis netlist retains this latch. Existing width-truncation warnings
cover an unused debug address, output-buffer address narrowing and the
documented 18-to-16-bit LIF stimulus conversion.
