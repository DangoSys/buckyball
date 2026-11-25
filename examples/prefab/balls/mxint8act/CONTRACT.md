# Mxint8actBall: packed MXINT8 activation

One instruction, `MXINT8ACT`: an integer activation, optionally multiplied by a
gate, on packed MXINT8 rows. It contains no floating-point arithmetic.
The source, table and destination are three distinct allocated one-group banks.

`rs1` uses the ordinary input/table/output bank fields and logical row count M.
Read-old and update flags are zero. `rs2` contains:

| Bits | Meaning |
|---|---|
| 0 | gated activation |
| 16:1 | output width K, positive and divisible by 32 |
| 30:17 | input base, in 128-bit bank rows |
| 44:31 | output base, in 128-bit bank rows |
| 58:45 | table base, in 128-bit bank rows |
| 63:59 | zero |

An ungated input row has K codes followed by K/32 E8M0 exponent bytes.
A gated input bank contains M packed activation rows followed by M packed gate
rows, each with K codes and K/32 exponent bytes. Output rows have K codes and K/32 exponents.
Each logical row is padded to 16 bytes. All output padding is zeroed.
Codes lie in [-127,127]; exponent byte 255 is illegal. Every addressed span
must fit its allocated bank.

The table has 4096 signed Q26 entries at x = -16 + index/128: 16 KiB, 1024
bank rows of 128 bits, all in one bank. A design instantiates this Ball only
with 128-bit banks of at least 1024 rows. The first calibrated table is GELU.
Table generation is offline. The high tail is identity, the low tail is zero,
and tiny blocks (input exponent below -24) use the linear term x/2. Interior
values interpolate neighboring table entries using integer multiply and
round-to-nearest-even. The upper endpoint of the final interval is 16 in Q26.

The intermediate value is Q16 relative to the input block scale. A gated
operation multiplies it by the gate's signed code and adds the gate exponent;
it does not requantize between activation and multiplication. For each output
block of 32 values, e_out is input_scale_exponent + floor(log2(max_abs_relative))
- 22, clamped to [-127,127]. A zero block uses e_out = 0. Codes are rounded
to nearest even and saturated to [-127,127].

Invalid encodings or storage spans fail. Source and table are read-only; output
cannot alias either. Thread-visible addresses are ordinary bank operands;
this instruction has no hart/device SHM semantics.

Verification proceeds through C/C++ ISA reference and BEMU, compiler/MLIR on
BEMU, short Verilator comparison, then UVM. Model adoption also requires the
existing Laya 200-record (1000-decision) accuracy loss to remain within one
percentage point of the FP32 baseline. The integer numerical prototype lives
in Poly Laya's `frontend/integer_activations.py`.
