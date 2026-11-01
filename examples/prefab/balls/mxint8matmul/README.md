# Mxint8matmulBall

`Mxint8matmulBall` multiplies packed MXINT8 rows into FP32:
`C[m, n] = A[m, k] . B[n, k]^T`, block by block in FP32, optionally onto
the old C.

```
MXINT8MM(a, b, c, m, n, k, a_base, b_base, c_base, accumulate)    funct7 0b100_xxxx
```

## Encoding

| field              | bits         | meaning                                 |
|--------------------|--------------|-----------------------------------------|
| rs1 `BB_BANK0`     | rs1[9:0]     | A bank                                  |
| rs1 `BB_BANK1`     | rs1[19:10]   | B bank                                  |
| rs1 `BB_BANK2`     | rs1[28:20]   | C bank                                  |
| rs1 `BB_READ_WR_OLD` | rs1[29]    | equal to `accumulate`                   |
| rs1 `BB_ITER`      | rs1[62:30]   | `k`                                     |
| rs2 `m`            | rs2[9:0]     | rows of A and C                         |
| rs2 `n`            | rs2[19:10]   | rows of B, columns of C                 |
| rs2 `a_base`       | rs2[33:20]   | bank row of A row 0                     |
| rs2 `b_base`       | rs2[47:34]   | bank row of B row 0                     |
| rs2 `c_base`       | rs2[61:48]   | bank row of C row 0                     |
| rs2 `accumulate`   | rs2[62]      | start from the old C instead of +0.0    |
| rs2[63]            | —            | must be 0                               |

The C macro is `bb_mxint8mm` in `workloads/isa/mxint8matmul.h`; the compiler
operations are `mxint8matmul_ball.bank_matmul_mxint8` (virtual banks, the old
C read when `accumulate`) and `mxint8matmul_ball.matmul_mxint8` (physical
banks), both with all bases 0.

## Layout

Banks have 128-bit rows (16 bytes) and one group (`col = 1`).

- A row `i` at bank row `a_base + i*S(k)`, B row `j` at `b_base + j*S(k)`,
  `S(k) = k/16 + ceil(k/512)`: the `k` INT8 codes, then the `k/32` exponent
  bytes `e + 127` (the packed rows `Mxint8quantizeBall` and `quant.py`
  write). Any INT8 code, `-128` included, is accepted.
- C row `i`: `n` little-endian FP32 at bank rows `c_base + i*n/4 ..`. Only the
  C window is written. Without the in-place update bit rs1[63] C is a new
  version of its bank: rows outside the window are undefined afterwards.

## Numerics

Per element `C[i][j]`:

```
acc = accumulate ? old C[i][j] : +0.0
for b in 0 .. k/32 (in order):
  p   = sum of codeA[i][32b+l] * codeB[j][32b+l], l < 32   (exact, |p| <= 2^19)
  s   = byteA[i][b] + byteB[j][b] - 254
  t   = fp32(p * 2^s)         rounded to nearest even (subnormals kept)
  acc = fp32(acc + t)         IEEE add, rounded to nearest even
C[i][j] = acc
```

`t` in integer form: `L = msb(|p|)`, `E = L + s + 127`; `E >= 1` gives the
exact normal `sign | E << 23 | ((|p| << (23 - L)) & 0x7fffff)`; `E <= 0`
gives `sign | RNE(|p| * 2^(s + 149))` subnormal units (a negative `p` that
rounds to zero gives `-0.0`; `p = 0` gives `+0.0`).

The block sums are exact, so a block's term equals the FP32 dot product of
the dequantized block whenever that product is representable; the only
rounding is the block-ordered FP32 accumulation. A product split over K
(several instructions with `accumulate`) continues the same order.

## Checks (fail loudly: BEMU panics, RTL asserts)

`m > 0`, `n > 0`, `n % 4 == 0`, `k > 0`, `k % 32 == 0`, A, B and C pairwise
distinct banks allocated with one group, `a_base + m*S(k)`, `b_base + n*S(k)`
and `c_base + m*n/4` within the bank, rs1[29] equal to `accumulate`, the old
C finite, every term and every sum finite, reserved fields zero.

## Hardware

Two read ports (A and the old C on port 0, B on port 1) and one write port. A
`rows x cols` array of PEs (`configs/default.toml`: 4 x 8) walks the output
tiles row-major; per tile and K block it reads the two code rows and the
exponent row of each A and B row, multiplies `lanes` code pairs per PE per
cycle (8), converts the block sum to its FP32 term and adds it with a
hardfloat adder (round to nearest even). Results do not depend on the
geometry.

## Tests

`workloads/ctests/vectors.py` writes `vectors.h` offline: A and B packed by
`quantize_mxint8` of `quant.py`, C by the loop above (`mm_small`: bases
1/2/3; `mm_accumulate`: onto an old C, tiles with remainders; `mm_extremes`:
subnormal and zero terms, cancelling blocks, terms near the top of FP32, a
block of `-128` codes with `p = 2^19`; `mm_long`: k = 544, two exponent rows,
bases 3/1/2, onto an old C). `mm_<case>.c` are the thread programs; a design
runs them and compares the C window with `<case>_expected` bit for bit.
`workloads/mlir_tests` holds the instruction encoding (`ball/`) and the
`mm_accumulate` case through the bank operation (`bank/`).
