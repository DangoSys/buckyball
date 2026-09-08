# NliBall

`NliBall` evaluates a non-linear function with **NLI (Non-uniform Linear
Interpolation)**: the function is approximated by a small set of piecewise-linear
segments whose cutpoints are placed *non-uniformly*, densely where the function
is most curved and sparsely where it is flat. See *"NLI: Non-uniform Linear
Interpolation Approximation of Nonlinear Operations for Efficient LLMs
Inference"*.

LLMs depend heavily on non-linear operators such as **SiLU** (Swish), GELU,
Softmax and RMSNorm. Linear layers are already well optimised; these non-linear
steps are the remaining bottleneck because they usually need high-precision
floating-point arithmetic. A uniform lookup table needs a huge table to reach
acceptable accuracy and still collapses on the extreme outliers common in LLM
activations. NLI fixes both: the cutpoints are found **offline**, so a tiny,
calibration-free segment table keeps high accuracy across the whole input range
— including outliers — in fixed-point hardware.

The Ball itself is **function-agnostic**: it holds no function constants. The
segment table (cutpoints / slopes / intercepts) is a runtime input loaded into a
coefficient bank with `MVIN`, so one `NliBall` approximates *any* non-linear
function (SiLU, GELU, exp, …) by swapping the table.

## Command

`NLI(input_bank, table_bank, output_bank, iter)`:

| operand   | meaning                                                    |
|-----------|------------------------------------------------------------|
| `input_bank`  | virtual bank holding INT8 input (16 lanes per 128-bit row) |
| `table_bank`  | virtual bank holding the segment table (one bank, `col=1`) |
| `output_bank` | virtual bank receiving INT8 output                          |
| `iter`    | number of 128-bit rows to process (`1 <= iter <= bankEntries`) |

The three banks must be distinct and each allocated with `col=1`. `rs2` is
reserved and must be zero. There is no fallback layout and no tail path.

## Data layout

Data rows mirror `LutBall`: one 128-bit bank row carries 16 signed INT8 lanes.

The coefficient bank is a *shared* (`col=1`) table of three rows:

| row | bytes   | content                                        |
|-----|---------|------------------------------------------------|
| 0   | 0 .. 14 | 15 interior cutpoints (INT8, ascending)        |
| 0   | 15      | unused                                         |
| 1   | 0 .. 15 | 16 segment slopes (INT8, Q7: value / 128)      |
| 2   | 0 .. 15 | 16 segment intercepts (INT8)                   |

The 15 cutpoints `c_1 < ... < c_15` tile the signed INT8 domain into 16
segments `[c_i, c_{i+1})` with `c_0 = -128`, `c_16 = 128`. Segment selection is
a comparator tree (`seg = #{j : x >= c_j}`).

## Datapath

For every input lane `x` the Ball computes, in one pass:

```text
seg = count of cutpoints <= x                 (16 non-uniform segments)
out = clamp( (slope[seg] * x) >> 7 + intercept[seg], -128, 127 )
```

`>> 7` is an **arithmetic** shift (floor toward -inf); `slope * x` is computed
in signed 16-bit. This exact semantics is shared bit-for-bit by the RTL
(`arch/src/main/scala/NliBall.scala`), the emulator
(`emu/src/nli.rs`) and the C reference model in the test.

## Example: SiLU (Swish)

`scripts/gen_silu_table.py` computes the offline NLI table for
`SiLU(x) = x / (1 + e^-x)` over the signed INT8 domain and emits the C constants
used by the test. The shipped table (16 segments) achieves:

```text
max absolute error = 1.0 LSB   (at x = 37)
mean absolute error = 0.50 LSB
```

over the full INT8 domain `[-128, 127]` — i.e. the approximation is already at
the fixed-point quantization floor, including the extreme outliers. This is the
point of NLI: a traditional **piecewise-constant** LUT needs **256 entries
(16x the storage)** to reach the same 1-LSB worst case, and a 16-entry uniform
LUT reaches only `max = 8.0 LSB / mean = 2.0 LSB` because its constant bins
collapse on the steep region near `x = 0`. 16 non-uniform *linear* segments get
the 256-entry accuracy with 16x less table. The all-zero intercepts are a
genuine property of SiLU (it is `~0` for `x << 0`, `~x` for `x >> 0`, and
crosses the origin with slope `~0.5`).

```text
cutpoints = {-96, -64, -44, -30, -20, -13, -8, -4, -1, 2, 5, 9, 16, 28, 48}
slopes    = {  0,   0,   0,   0,   0,   0,  0, -2, -8, 87, 127, 127, ... 127}
intercepts= {  0,   0,   0,   0,   0,   0,  0,  0,  0,  0,   0,   0, ...   0}
```

Regenerate the table:

```sh
python3 examples/balls/nli/scripts/gen_silu_table.py
```

## Test

`workloads/ctests/nli_test.c` loads a 48-byte SiLU table and 64 inputs
(including extreme outliers), issues one `NLI` command, then checks:

1. **Datapath** — exact match against an independent C reference of the
   segment-select + Q7 MAC contract;
2. **Accuracy** — the result matches `round(SiLU(x))` within `±1` LSB.

## Layout of the ball

```text
examples/balls/nli/
├── arch/src/main/scala/NliBall.scala        # Chisel RTL
├── compiler/src/Dialect/Buckyball/NliBall.td
├── compiler/src/Dialect/Buckyball/Transforms/LegalizeForLLVMExport.cpp
├── compiler/src/Conversion/LowerBuckyball/AssignPhysicalBankPatterns.cpp
├── configs/default.toml
├── emu/src/{lib.rs,nli.rs}                  # BEMU golden model
├── scripts/gen_silu_table.py                # offline cutpoint optimisation
└── workloads/{isa/nli.h, ctests/nli_test.c}
