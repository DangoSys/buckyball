# Im2col Ball

This directory implements the Buckyball `Im2colBall` RTL for instruction
function 48.

## ISA contract

```c
bb_im2col(input_bank, output_bank, iter, ksize, stride, padding)
```

- `iter` is both the input height and input width.
- `ksize` is both the kernel height and kernel width.
- `stride` applies to output rows and columns.
- `padding` is zero-padding on every input edge.
- Input and output elements are signed or unsigned 8-bit payload bytes; Im2col
  only rearranges bytes and does not multiply or accumulate them.

The output dimension is:

```text
outputDim = floor((iter + 2 * padding - ksize) / stride) + 1
```

Windows are emitted in output-row-major order. Each window is flattened in
kernel-row-major order, so the total output byte count is:

```text
outputDim * outputDim * ksize * ksize
```

For output coordinate `(orow, ocol)` and kernel coordinate `(krow, kcol)`, the
unpadded input coordinate is:

```text
inputRow = orow * stride + krow - padding
inputCol = ocol * stride + kcol - padding
```

An out-of-range coordinate emits zero. Otherwise the source byte offset is:

```text
inputRow * iter + inputCol
```

## Files

- `Im2col.scala`: top-level command and response control.
- `Im2colConfigRegs.scala`: ISA field decoding and validation.
- `Im2colWindow.scala`: output-window and kernel traversal.
- `LineBufferManager.scala`: input preload and padded element selection.
- `StreamWriter.scala`: contiguous output packing and bank writes.
- `Im2colBall.scala`: Blink-compatible ball wrapper.
- `configs/Im2colBallParam.scala`: maximum dimension and element width.
