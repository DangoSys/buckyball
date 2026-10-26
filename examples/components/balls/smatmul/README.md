# SMatMulBall

`SMatMulBall` computes signed INT8 matrix products with signed INT32 output.
The array is 16x16, SRAM rows are 128 bits, and every SRAM is single-port.

## Commands

Every command uses the framework-defined `rs1` layout: bank0 in bits 9:0,
bank1 in bits 19:10, bank2 in bits 29:20, and `iter` in bits 63:30. There are
no bank-relative base fields; every operand starts at row zero.

`SMATMUL_BIAS` reads four rows from bank0 and retains one 16-element INT32
bias vector. bank1, bank2, and `rs2` are zero, and `iter` is four.

`SMATMUL_OS` uses bank0 for A, bank1 for B, and bank2 for C. `iter` is K.
`rs2[11:0]` is M, `rs2[23:12]` is N, and bits 24 and 25 are the first/last
accumulation-block flags and bits 31:26 the C base row. The remaining `rs2`
bits are zero. M is one or a positive multiple of tileRows, N is exactly 16.

A and B have the same number of groups G. Group g holds reduction block g in
the single-group layouts below, so `iter` is K per group (a positive multiple
of 16) and the instruction accumulates over `K = iter * G`, walking group 0 to
group G-1 for every output tile.

## Layout

Every result row has four 128-bit words. With `outBW=1`, logical output row
`r` uses four consecutive bank rows:

```text
line r*4 + word
```

C rows continue across groups: line `L = base + r*4 + word` is written to
group `L / bankEntries`, line `L % bankEntries`. M is limited by the
accumulator (`max(bankEntries / 4, 32)` rows).

The Ball keeps partial sums internally between commands with `last=0`; only
the command with `last=1` writes C. A later accumulation block may select new
A and B banks, but each still starts at bank row zero.
