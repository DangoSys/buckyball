# Int2FpBall UVM

DPI reference lives in `casegen/` and reuses `../emu/src/model.rs`.

The full SystemVerilog UVM environment is the next verification slice. It can
mirror `examples/balls/fp2int/verify/` with a directed `INT32 -> FP32` case
using `op1_col = 1` and `wr_col = 1`.

Build DPI:

```console
cargo build --manifest-path casegen/Cargo.toml
```
