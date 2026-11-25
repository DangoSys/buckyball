# LAYERNORM Ball UVM

```
nix develop -c bbdev uvm --run '--ball piplus/balls/layernorm'
```

`+LAYERNORM_CASE=<n>` on the simv command line runs one case.

- `casegen/` (Rust, DPI): 23 directed and 40 random LAYERNORM instructions.
  `reference.rs` is the BEMU model's arithmetic (`emu/src/layernorm.rs`,
  `workloads/isa/layernorm.h`) in FP32 round to nearest even, so the output
  is compared bit for bit. Each case gives the instruction fields and three
  1024-row bank images: input, parameters, expected output. Rows and lanes the
  Ball must not use hold quiet NaNs; the output bank outside the window holds
  `0x5a` guards.
- `src/layernorm_pkg.sv`: the sequence, the scoreboard and the covergroup.
  The shared bank model holds a read response until the Ball accepts it
  (the Ball stalls them in pass 2).
- The scoreboard checks every read address per port in order (port 0: x on
  each pass, port 1: eps once, then gamma/beta per bank row), every write
  (bank, group, ROB id, row, full mask, data), the completion, and that
  inputs, parameters and the output guards are unchanged. All cases run
  back to back after one reset.
- Coverage: W bins (16 to 2048, including the LayerNorm and RMSNorm maxima),
  W mod 4, rms, rows (1 to 256), value classes (random, constant, subnormal,
  large offset, signed zeros, wide exponents, eps 0, large eps), windows that
  end on the last bank row, zero/nonzero bases, completion backpressure,
  crosses rms x tail and rms x class.
