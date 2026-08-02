# Int2FpBall UVM

DPI reference lives in `casegen/` and reuses `../emu/src/model.rs`.

Full SystemVerilog UVM env is the next slice after Fp2Int UVM INT8 landing:
mirror `examples/balls/fp2int/verify/` with directed cases for:

1. INT32→FP32 `(op1_col=1, wr_col=1, output_mode=0)`
2. INT32→INT8 `(op1_col=4, wr_col=1, output_mode=1)`

Build DPI:

```console
cargo build --manifest-path casegen/Cargo.toml
```
