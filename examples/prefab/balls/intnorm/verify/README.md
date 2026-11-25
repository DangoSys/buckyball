# IntnormBall UVM Verification

Verifies the generated `IntnormBall` (INTNORM: FP32 rows -> Q16 LayerNorm ->
packed MXINT8) against a reference ported from the BEMU model
(`emu/src/norm.rs`), using the shared Blink UVM framework.

## Structure

- `casegen/src/reference.rs`: the INTNORM reference over byte images of the
  three banks; it also counts the rounding/clipping events coverage needs
- `casegen/src/cases.rs`: directed and random cases and the DPI API
- `src/intnorm_pkg.sv`: sequence, scoreboard with covergroups, env and
  `intnorm_ball_test`
- `src/tb_top.sv`: `bb_blink_if#(2,1)` + `IntnormBall`
- `filelists/intnorm_ball.f`: VCS filelist (`@UVM@` / `@RTL@`)

## Test plan

`intnorm_ball_test` resets once and runs every case back to back, with a
random ROB id (within `BB_ROB_ID_W`) and random distinct banks and groups:

| case | what it covers |
| --- | --- |
| `round` | `workloads/ctests/intnorm_round.cpp`, its codes are asserted in casegen |
| `zero` | ±0, subnormals and underflow inputs; result = beta; zero and narrow blocks |
| `extreme` | ±32767.99, -32768.0, then every FP32 exponent class |
| `input_ties` | K=1536, inputs exactly halfway between Q16 steps |
| `beta_codes` | constant rows (z = 0): code ties, clip to 127, 32-bit blocks |
| `capacity` | K=512 with input, coefficients and output+scratch ending at the last bank row |
| `k1024`, `k1536` | wider rows, nonzero bases, K=1536 at the bank ends |
| `epsilon` | epsilon above 2^62 and full-range gamma |
| `affine` | negative gamma, beta near the Q16 limits |
| `random10..22` | random K, rows, bases, row scales and means, gamma, beta, epsilon; `random10` holds `cmdResp.ready` low 400 cycles |

The scoreboard preloads every bank row (NaN inputs and guard bytes outside the
footprint) and checks:

- every read is inside the input, coefficient or scratch footprint;
- every scratch write equals the Q16 results of the current row;
- every packed output row is written once and equals the reference;
- the completion carries the ROB id, after exactly `rows*K/4` scratch and
  `rows*(K/16+K/512)` output writes; input and coefficient banks are unchanged
  and the whole output bank equals the reference image.

Coverage (`intnorm_scoreboard`): K x rows, bank-end footprints, zero/nonzero
bases, epsilon width and dominance, negative means, input and code ties,
clips, zero/narrow/32-bit blocks, completion backpressure, FP32 exponent class
x sign of every input, and every MXINT8 code and exponent byte written.

## Run

```console
bbdev uvm --run '--ball intnorm'
```

The Ball runs alone from `configs/default.toml` (ball id 0); bbdev passes
`+BID=0` and `BB_VERIFY_CONFIG` (bank rows, ball id).
