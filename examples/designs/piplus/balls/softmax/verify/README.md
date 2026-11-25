# SOFTMAX Ball verification

`nix develop -c bbdev uvm --run '--ball piplus/balls/softmax'` runs the Ball
alone (configs/default.toml) against `softmax_ball_test`.

- `casegen/` (Rust, DPI): 50 cases, 0 to 17 directed and 18 to 49 random. Each
  case gives the command fields, the input bank image and the expected output
  bank image. `reference.rs` copies the FP32 operation order of
  emu/src/softmax.rs (workloads/isa/softmax.h): the RTL must match bit for bit.
- `src/softmax_pkg.sv`: the sequence, scoreboard, coverage and test.
  - Every read is checked against the three passes per row (maximum, sum,
    outputs) in bank-row order, with bank, group and ROB id.
  - Every write is compared bit-exactly with the expected bank row, mask all
    ones. On completion both bank images are compared whole: the input is
    unchanged and the rows outside the output window keep their guard bytes.
  - Cases run back to back after one reset. Case 16 holds `cmdResp.ready` low
    and checks that the completion stays and no command is accepted.
  - Coverage: W tail and range, rows, scale_log2 (-126, 127, signs), window
    bases and last bank rows, ROB ids, the semantic events the reference
    counts (exponential underflow, -inf, x - m overflowing to -inf, ties of the
    maximum, subnormal x and y, u near -126), the input and output classes of
    every element, and the inf/NaN/-inf lanes after W that must be ignored.
