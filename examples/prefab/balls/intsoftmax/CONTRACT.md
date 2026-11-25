# Integer attention softmax

IntSoftmaxBall uses two read banks and one write bank, each one group of
16 KiB. The three allocated banks are distinct. The table occupies all
1024 128-bit rows of its bank. No floating point arithmetic is executed.

`rs1` carries the ordinary input, table, output banks and M logical rows.
The typed emitter clears read-old and update; their lifecycle semantics belong
to the common frontend, and the Ball reads no old destination. `rs2[7:0]` is K, between 1 and 128;
`rs2[63:8]` is zero. Each input/output row occupies ceil(K/4) bank rows,
starting at bank row zero. M is positive and M*ceil(K/4) fits one bank.
Input words are F32 encodings. Output words are exact Q15 probabilities
in F32 representation. Unused output lanes in the final bank row are zero.

For each logical row:

1. Clamp scores to [-4096,4096] and convert to signed Q8 with RNE. Infinity
   saturates; NaN is illegal. This conversion uses integer bit operations.
2. Subtract each score from the integer maximum and clamp to [0,4095].
3. Read I32 table entries round(exp(-i/256)*32768). Entry zero is 32768;
   all entries lie in [0,32768]. The table is generated and verified offline;
   execution checks the entries it accesses without scanning unused entries.
4. Let s be the sum and w an entry. Compute floor((w*32768+floor(s/2))/s).
5. Encode that unsigned Q15 value as F32 using integer bit operations.

One reciprocal r=floor(2^32/s) suffices per row. For n=w*32768+floor(s/2),
q0=floor(n*r/2^32) is exact or one low because n<2^31. Increment q0 when
n-q0*s >= s. This is exactly the division above, not an approximation.

The instruction reads no old destination and defines its entire output
span. It has no hart/device SHM semantics. Invalid encodings and bank spans
fail. Validation follows C/BEMU, typed MLIR/BEMU, small RTL, then UVM and
resource checks. The existing Laya 200-record/1000-decision Q8/Q15 oracle
scores 43.1% against a 42.6% FP32 baseline; hardware must match that oracle.
