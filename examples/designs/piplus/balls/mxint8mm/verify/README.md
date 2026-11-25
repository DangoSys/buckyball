# Mxint8mmBall UVM

```
nix develop -c bbdev uvm --run '--ball piplus/balls/mxint8mm'
```

The Ball runs alone (`configs/default.toml`, ball id 0) against the common
BLink agents of `verify/uvm`. `casegen` (Rust, loaded through DPI) holds the
reference model, a port of `emu/src/matmul.rs`, and the cases.

## Cases

Each case is one or more commands back to back on one reset. The generator
keeps an image per (bank, group) that mirrors the testbench memory, so that a
later command accumulates onto the C an earlier one wrote. Only the bank rows
a command may read are preloaded: a read anywhere else stops the memory model.

- 0-15 directed: one tile; partial tiles with one column group; accumulate onto
  a renamed destination with gaps between C rows; bias only; accumulate and
  bias in place; 16 blocks on one exponent row and 17 blocks on two; extreme
  codes and exponent bytes (-128, 0, 255); subnormal and flushed terms onto
  -0.0 and subnormal start values; large terms; every term 2^127 (all codes
  -128); blocks whose products cancel; a three-command chain (renamed, then in
  place, odd block counts); a response held back by `cmdResp.ready`; A and C
  windows ending at the last bank row; the widest B the bank holds.
- 16: every start value (old C, bias, both; renamed and in place) with one block.
- 17-30 random shapes, layouts, banks, groups, start values and value modes.

## Checks

- The command reaches the Ball unchanged.
- Reads: port 0 reads A (op1 bank and group) or, with accumulate or bias, the
  old C (`wr_read_bank`, wr group), port 1 reads B, all with the command's ROB id.
- Writes: each C bank row the reference writes is written once, to the wr bank
  and group with a full mask and the exact FP32 data; nothing else is written.
- The response carries the ROB id and comes after all writes, and it stays
  valid while held back.

## Coverage

`cov/mxint8mm_cov.svh` must reach 100%: start value and destination, M and N
against the tile (`rows`, `cols` from `parameters.toml`), K blocks against the
shared exponent row, C pitch, A base, chained commands, held response, bank
end, the reference's numeric events (subnormal, flushed and huge terms,
cancelled blocks, code -128, exponent bytes 0 and 255, inexact additions,
special start values), and start value x one block or a few.
