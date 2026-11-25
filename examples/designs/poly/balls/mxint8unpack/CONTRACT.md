# Packed MXINT8 bit unpack

`mxint8unpack`, class `examples.designs.poly.balls.mxint8unpack.Mxint8UnpackBall`,
uses one read bank and one write bank. `MXINT8UNPACK` has funct7 50 (1R1W).
The ordinary source is bank0 and destination is bank2. Bank1 is not an
operand: emitters set it to zero and execution ignores its encoding.
`iter` is B, the number of complete 32-code blocks, in [1,128]. `rs2` is zero.
The typed emitter clears read-old and update; the Ball reads no old output.
Both allocated banks have one group, 1024 rows of 128 bits (16 KiB), and are
separate. All addresses start at row zero; there are no base fields or DDR access.

The source contains 32*B signed I8 codes followed by B unsigned I8 exponents.
Code block b occupies rows 2*b and 2*b+1. Its exponent is byte b%16 of source
row 2*B+b/16. Unused bytes after the exponent array are ignored. Output block b
occupies eight consecutive rows starting at 8*b, with four little-endian IEEE32
words per row. Every output byte is defined. The Ball requests no writes beyond 128*B;
bytes outside that result span are not part of the produced bank state.

For code q and raw exponent e, output encodes exactly q*2^(e-127), with IEEE32
infinity on overflow. This is the bit-identical boundary of the existing MXINT8
GP decode; the Ball performs only integer operations:

- q=0 gives positive zero, for every e including 255.
- Let a=abs(q), computed wide enough to represent abs(-128)=128, and l=floor(log2(a)).
- Sign is q<0. For e+l>=255, output is signed infinity.
- The only nonzero subnormal case is e=0 and a=1: sign | 0x00400000.
- Otherwise exponent field is e+l; mantissa is (a<<(23-l)) & 0x007fffff.

All 256 code bit patterns and all 256 exponent values are legal. No rounding,
lookup table, FP arithmetic, saturation of finite values, quantization, or change
to downstream ordered accumulation is introduced. Typed physical lowering
requires different source/destination banks; BLink checks the received renamed
bank identities. Invalid rs2, zero or oversized B and unallocated or multi-group
banks fail at their owning interface.

C/header and BEMU pin this contract first. BankSSA/physical typed operations and
MLIR execution follow only after C/BEMU passes. RTL follows only after that gate;
UVM/resources follow the small RTL gate. The calibrated single-group BEMU
operation latency is 11+24*B+3*ceil(B/16) cycles: B=1/8/17/128 measured
38/206/425/3107 cycles in RTL. This calibration is not a hardware speedup.
