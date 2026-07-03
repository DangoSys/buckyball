# Im2col Ball

This directory implements the Buckyball `Im2colBall` RTL.

## Files

- `Im2col.scala`: top-level command, progress, and response wiring.
- `Im2colConfigRegs.scala`: command field decode and argument validation.
- `Im2colWindow.scala`: row-major output window traversal.
- `LineLoadCtrl.scala`: line-buffer read request sequencing.
- `LineBufferManager.scala`: line-buffer storage and element lane selection.
- `StreamWriter.scala`: element packing and bank write requests.
- `FIFO.scala`: physical line-buffer slot rotation.
- `Im2colBall.scala`: ball wrapper.
- `configs/Im2colBallParam.scala`: local width and kernel-size parameters.

## Behavior

The implementation follows the BEMU `im2col` layout: input bytes are addressed
as a continuous row-major byte stream. For a requested element, the source byte
offset is:

```text
(row + krow) * inCol + col + kcol
```

The line loader reads enough bank beats to cover each image row, including rows
whose byte start is not beat-aligned. The element selector uses the row byte
offset and column pointer to pick the correct lane from the buffered beats.

The top level does not use an explicit phase `Enum`/`switch` state machine.
Operation progress is represented by handshake flags such as `running`,
`linesReady`, `finishPending`, and `respPending`.
