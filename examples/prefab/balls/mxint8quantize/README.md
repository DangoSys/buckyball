# Mxint8quantizeBall

`Mxint8quantizeBall` quantizes rows of FP32 values to packed MXINT8 rows,
bit for bit as `quantize_mxint8` of `stack/compiler/frontend/quant.py`.

```
MXINT8QUANT(input, output, rows, k, in_base, out_base)    funct7 0b011_xxxx
```

## Encoding

| field            | bits         | meaning                                   |
|------------------|--------------|-------------------------------------------|
| rs1 `BB_BANK0`   | rs1[9:0]     | input bank                                |
| rs1 `BB_BANK2`   | rs1[28:20]   | output bank                               |
| rs1 `BB_ITER`    | rs1[62:30]   | `rows`                                    |
| rs1[19:10], rs1[29] | —         | must be 0                                 |
| rs2 `k`          | rs2[15:0]    | values per row                            |
| rs2 `in_base`    | rs2[29:16]   | bank row of input row 0                   |
| rs2 `out_base`   | rs2[43:30]   | bank row of output row 0                  |
| rs2[63:44]       | —            | must be 0                                 |

The C macro is `bb_mxint8quant` in `workloads/isa/mxint8quantize.h`; the
compiler operations are `mxint8quantize_ball.bank_quant_mxint8` (virtual
banks) and `mxint8quantize_ball.quant_mxint8` (physical banks), both with
`in_base = out_base = 0`.

## Layout

Banks have 128-bit rows (16 bytes) and one group (`col = 1`).

- Input row `r`: `k` little-endian FP32 at bank rows `in_base + r*k/4 ..`.
- Output row `r` starts at bank row `out_base + r*S(k)`,
  `S(k) = k/16 + ceil(k/512)`: bytes `[0, k)` the INT8 codes, bytes
  `[k, k + k/32)` the exponent bytes `e + 127`, zero up to `S(k)*16` bytes.
  Exponent byte `b` of a row is byte `b % 16` of the row's bank row
  `k/16 + b/16`.
- The output is written whole (codes, exponents and zero padding). Without
  the in-place update bit rs1[63] the output is a new version of its bank:
  rows outside the packed rows are undefined afterwards.

## Numerics

Per row and block of 32 values `x` (FP32 bits):

- `max` = the largest `bits & 0x7fffffff` (an integer compare).
- `e = 0` if `max == 0`; else `e = max(F - 133, -127)` with `F = max >> 23`
  (`F = 0`, a subnormal maximum, gives `-127`). This is
  `floor(log2(max |x|)) - 6`, at least `-127`.
- `code(x) = clamp(round_half_even(x / 2^e), -127, 127)`, in integer form:
  `f = (bits >> 23) & 0xff`, `sig = (bits & 0x7fffff) | (f ? 0x800000 : 0)`,
  `sh = 150 + e - max(f, 1)` (always `>= 17`); magnitude `0` if `sh > 31`,
  else `sig >> sh` rounded to nearest even on the dropped bits, at most 127;
  negated for a negative `x`. `-0.0` gives code 0.

## Checks (fail loudly: BEMU panics, RTL asserts)

`rows > 0`, `k > 0`, `k % 32 == 0`, input bank != output bank, both banks
allocated with one group, `in_base + rows*k/4 <= entries`,
`out_base + rows*S(k) <= entries`, every input value finite, reserved fields
zero.

## Hardware

One read port and one write port. Per block: eight bank reads, the block
exponent, two code rows written by 16 parallel encoders, and at every 16th
or the row's last block the zero-padded exponent row. No parameters.

## Tests

`workloads/ctests/vectors.py` writes `vectors.h` offline from
`quantize_mxint8` (`quant_edges`: ties, clamps, signed zeros, subnormals,
all-zero and tiny-normal blocks, the FP32 extremes; `quant_random`: 3 rows of
544 values, two exponent rows per output row). `quant_edges.c` and
`quant_random.c` are the thread programs; a design runs them and compares the
packed rows with `<case>_expected` byte for byte.
`workloads/mlir_tests` holds the instruction encoding (`ball/`) and the
`quant_random` case through the bank operation (`bank/`).
