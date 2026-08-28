# SMatMulBall

`SMatMulBall` computes signed INT8 matrix products with signed INT32 output.
The array is 16x16, SRAM rows are 128 bits, and every SRAM is single-port.

## Commands

Both commands use `iter` for three bank-relative base rows: A, B, then C.
The remaining `iter` bits and `rs2[63:36]` must be zero. A, B, and C must be
different allocated virtual banks.

`SMATMUL_OS` is output-stationary. `rows` and `k` are positive multiples of
16, `cols` is exactly 16. A and B use `cols=1`; C uses the SMatMulBall
`outBW` pbank groups from the chip configuration. A and B are provided as
16-wide K tiles; the Ball accumulates all K tiles before writing C.

`SMATMUL_WS` is weight-stationary. It computes one `16x16` A tile against a
`16xN` B tile. A and B use `cols=1`; B is panel-major in one pbank. C uses the
SMatMulBall `outBW` pbank groups from the chip configuration. A is read once
and remains resident while B panels are processed in increasing line order.

## Layout

Let `rounds = 4 / outBW`. Every result row has four 128-bit words. Word
`w` is written to C pbank `group = w % outBW` at `round = w / outBW`.

For OS, logical output row `r` uses `rounds` consecutive physical lines:

```text
line r*rounds + round
```

For WS, panel `p`, row `r`, and write round use line
`base + p*(16*rounds) + r*rounds + round`. With Pebble `outBW=2`, C group 0
holds columns `0..3` then `8..11` on adjacent lines; C group 1 holds columns
`4..7` then `12..15`. The two groups are written together in two rounds.

The Ball owns all three virtual banks until completion. It does not interleave
rows from different groups, and it does not special-case padding; software
must provide 16-aligned shapes.
