# MXINT5GEMV

This experimental Ball is selected by `ToyInt5GemvConfig` and the isolated
`PolyGemv5Config`. Production Poly configurations are unchanged. It evaluates
one MXINT8 activation row against 32 scaled INT5 weight rows and emits MXINT8.
All physical banks are 16 KiB, with 128-bit rows. Four B groups each hold eight
output rows; A and C each use one group. A, B and C are distinct logical banks.

## Instruction

The generated design instruction table is authoritative for the opcode.
`op1_base`, `op2_base`, `wr_base`, `iter` and `read_wr_old` are the standard
bank operands, K and old-C selection. No other user rs1 fields are read.
The user rs2 fields are A code row [13:0], A exponent row [27:14], B base row
[41:28], C base row [55:42], accumulate [56] and finalize [57]. Bits [63:58]
are zero. Accumulate equals `read_wr_old`. The standard update field
rs1[63] selects an in-place C update. It does not change the dot-product,
accumulation or finalization arithmetic.

K is positive and divisible by 32. Total accumulated K is at most 32768.
A contains K signed INT8 codes followed through the separate exponent address
by K/32 E8M0 bytes. Both halves must fit their allocated bank.

Each B row contains K*5/8 code bytes, rounded up to a 16-byte boundary, then
ceil(K/512) factor rows and ceil(K/512) exponent rows. Each consecutive group
of eight codes occupies five little-endian bytes: code i uses bits [5i+4:5i].
Codes are signed two's complement in [-15,15]; -16 is invalid. Factors are
unsigned integers in [1,137]. Exponent 255 is invalid for either operand.
Padding is not read as an element. The complete eight-row B window in each
group and the 17-row C state window must fit the bank.

## Arithmetic and state

For each 32-element block and each output, compute the exact integer dot
product of A codes and B codes, then multiply that integer by the B factor.
Its power-of-two exponent is Ea + Eb - 254 - 4. There is no intermediate
INT8 rounding and no FP32 arithmetic. This is the native scaled INT5 numerical
weight variant qualified by `qwen8-native-five-full-calibration.json`.
That calibration qualifies the compressed weights, not arbitrary additional
output quantization. Compiler fusion must consume an existing explicit MXINT8
output boundary, or a recipe must separately qualify the new boundary.

Accumulate signed INT64 values with 28 guard bits relative to the largest
nonzero block exponent seen so far. Raising that common exponent uses signed
round-to-nearest, ties-to-even. Align each block term with the same rounding.
The state magnitude is at most `processed_blocks << 52` and must not exceed
INT64. Old state is consumed only when accumulate is set.

The 16-byte state header is processed-block count (u16), common exponent
(i16), valid (u8), weight code width 5 (u8), guard width 28 (u8), and nine zero
bytes. A valid common exponent is in [-258,250]. An invalid state has zero
exponent and all zero values. The header is followed by 32 little-endian i64
values. Partial output writes exactly 17 rows. A final output writes exactly
three rows: 32 signed MXINT8 codes, one E8M0 byte and 15 zero padding bytes.
When update is clear, C rows outside the write footprint are unspecified.
When update is set, the frontend retains the current physical C mapping and
serializes its readers and writers; all rows outside the 17-row partial or
three-row final footprint retain their previous bytes. Multiple states may
therefore share C at distinct non-overlapping bases. The 17-row capacity
check applies even to finalization. `read_wr_old` still describes whether
the computation consumes the selected old state; update also preserves
other live states when accumulate is clear.

For finalization, choose `max(-127, floor(log2(max_abs_value)) + scale - 34)`
as the MXINT8 exponent, capped at 127. Round values to nearest even and clamp
codes to [-127,127]. All-zero output has exponent zero, encoded as 127.
Invalid fields, operands, state, allocations or capacity fail immediately.

## Consumers and gates

The ISA header, C fixtures and BEMU implementation define Stage 1. The typed
bank/ball ops and MLIR execution tests are Stage 2. Both ops carry C base
as an i64 SSA operand and an explicit update flag. Lowering must prove
that C base is in [0,1007], including loop-generated bases. The standard
update bit preserves the physical mapping; it does not change the numeric
accumulate/read-old field. RTL follows only after
those gates pass, then small Verilator tests, UVM and resource measurements.
The C and bank/physical MLIR gates pass on BEMU and small RTL tests, including
K=32/512/1024, old-state accumulation, metadata refresh, mixed values and zero
blocks. The timing model is calibrated against those RTL measurements. The
operand prefetch reuses an overlapping physical bank row between adjacent
20-byte code blocks. UVM and resource qualification remain separate gates.
Production Poly configurations are unchanged.
