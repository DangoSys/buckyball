# Im2col Ball

This directory implements the Buckyball `Im2colBall` RTL for instruction
function 48.

## ISA contract

```c
bb_im2col(input_bank, output_bank, input_size, kernel, stride, padding,
          input_base, start_row, start_col, window_start, window_count,
          tile_rows)
```

- `rs1`: bank0 is the input, bank2 the output, `iter` the input height and
  width.
- `rs2`: `[3:0]` kernel, `[7:4]` stride, `[11:8]` padding, `[15:12]`
  start_col, `[19:16]` start_row, `[25:20]` input_base, `[35:26]`
  window_start, `[45:36]` window_count, `[50:46]` tile_rows (1, 2, 4, 8, or
  16); `[63:51]` are zero.
- Im2col only moves INT8 bytes; it does not multiply or accumulate.

## Layout

Every group of the input virtual bank is one channel block: group `g` line
`input_base + y * input_size + x` holds channels `16g .. 16g + 15` of pixel
`(y, x)`. With `Cg` input groups, `C = 16 * Cg`.

The output is SMatMulBall operand A for `M = window_count` windows and
`K = kernel * kernel * C`, ordered `(kernel row, kernel column, channel)`:

```text
outputRows = (input_size + 2 * padding - kernel - start_row) / stride + 1
outputCols = (input_size + 2 * padding - kernel - start_col) / stride + 1
window w   = output pixel window_start + w (row-major)
K tile t   = (kr * kernel + kc) * Cg + g        // one whole input row
K_TILES    = kernel * kernel * Cg
G          = K_TILES / output_groups            // output groups divide K_TILES
```

Output group `j` holds K tiles `j * G .. j * G + G - 1`, in the SMatMulBall A
layout with `tile_rows`-row M tiles:

```text
line(w, t) = ((w / tile_rows) * G + t % G) * tile_rows + w % tile_rows
```

A tile row is the input row of pixel
`(oy * stride + kr + start_row - padding, ox * stride + kc + start_col - padding)`,
or zero when that pixel is in the padding. Windows from `window_count` to the
next multiple of `tile_rows` are zero. Each output group therefore is the
SMatMulBall reduction block `j` with `k = 16 * G`.

## Files

- `Im2col.scala`: command decode, shape division, and the row-copy loop.
- `RestoringDiv.scala`: shared sequential divider.
- `Im2colBall.scala`: Blink-compatible ball wrapper.
- `configs/Im2colBallParam.scala`: maximum dimension and element width.
