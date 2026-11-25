# MXINT8 GEMV

Status: stages 0 through 3 passed C golden, compiler lowering, BEMU and small
RTL checks. Eight C fixtures cover 2288 output/guard bytes; two MLIR fixtures
cover 672 bytes. Traces confirm each partial writes 17 rows and each finalize
writes only 3 rows. Four-bank next-block prefetch preserves all ten fixtures. The current 128-MAC
array passes the same fixtures, including exact partial/final write footprints:
K512 partial is 399 cycles and K1024 finalize is 717 cycles. The frozen 256-MAC
prefetch version takes 395 and 713 cycles, respectively. Halving the multiplier
count costs four cycles per instruction; total LUT/DSP/area is not yet measured.
These are Ball latencies, not whole-model speedups. Production model mapping,
accuracy calibration and PPA/UVM are not complete.

## Execution and storage

One instruction computes 32 output channels of one input row. The array has
1 x 32 PEs, each multiplying 4 INT8 pairs per cycle: 128 MACs per cycle.
Four B banks can supply at most 64 INT8 weights per cycle. Accumulation and output quantization
use integers. This Ball contains no floating-point converter or adder.

Every physical bank remains 16 KiB with 128-bit rows. A and C each use one group;
B uses four groups, hence four independent physical banks. All three virtual
banks differ. Read channel 0 accesses A or old C; channels 1 through 4 access
B groups 0 through 3. There is one C write channel.

Let P(K) = K/16 + ceil(K/512), in bank rows. A codes and exponent bytes have
independent row bases. B group g stores eight consecutive packed MXINT8 rows,
for output channels 8g through 8g+7. Their row starts are b_base + j*P(K).
Within each B row, K code bytes precede K/32 exponent bytes and zero padding
to 16 bytes. Input codes are signed INT8; exponents are biased by 127 and must
be in [0,254]. K is positive and divisible by 32. All A code/exponent ranges,
eight B rows per group, and 17 C rows must fit their allocated banks.

## ISA

The mnemonic is MXINT8GEMV. Its funct7 comes from the design.pb instruction
declaration; C headers and backends do not embed a numeric opcode.

rs1 uses the framework bank fields: bank0=A, bank1=B, bank2=C, ITER=K.
READ_WR_OLD must equal accumulate; UPDATE and all other unused fields are zero.

rs2 fields:

| Bits | Meaning |
| --- | --- |
| 13:0 | A code base, in 128-bit rows |
| 27:14 | A exponent base, in 128-bit rows |
| 41:28 | B base, shared row offset in all four groups |
| 55:42 | C base, in 128-bit rows |
| 56 | accumulate: read the previous integer state |
| 57 | finalize: write MXINT8 instead of integer state |
| 63:58 | reserved, zero |

accumulate=false starts a fresh sum. accumulate=true consumes a state previously
written by this instruction. A sequence processes K segments in increasing
logical K order. Its total K must not exceed 32768 (1024 blocks).
Accumulation reads the old C version and writes the new C version through the
framework destination interface. Their physical banks need not be equal.

## Integer numerical rule

For each K/32 block, compute the 32 exact signed dot products p[j] using INT32.
Let s[j] = A_exp + B_exp[j] - 254. Terms with p[j]=0 do not choose an exponent.
The shared state holds 32 INT64 accumulators v[j], an exponent S, and a scale-valid
bit. Real values are represented by v[j] * 2^(S-32).

Process blocks in source order. For each block with a nonzero dot, select the
larger of the previous valid S and max(s[j] for nonzero p[j]). Right-shift previous
v[j] with signed round-to-nearest-even when S increases. Add p[j] scaled by
2^(32+s[j]-S), using the same signed rounding for right shifts. A right shift
of at least 64 yields zero. If all accumulators become zero, clear scale-valid
and set S to zero; subsequent nonzero blocks select a fresh scale.

The magnitude of each aligned term is at most 2^51. The total-K bound therefore
keeps each accumulator at or below 2^61 in magnitude, with signed INT64 headroom.
There is no saturation of the integer state and no intermediate INT8 quantization.

On finalize, let r=floor(log2(max(abs(v[j])))). For nonzero output, choose
q=clamp(r+S-32-6,-127,127). Output codes are signed RNE(v[j]*2^(S-32-q)), clamped
to [-127,127], with exponent byte q+127. All-zero output uses zero codes and
exponent 127. This is a declared quantization boundary; it does not claim bitwise
identity with the old FP32 block accumulation.

## C layout and footprints

Partial state occupies 17 rows: a 16-byte header followed by 32 little-endian
INT64 values. The header contains processed_blocks (u16 at byte 0), S (i16 at
byte 2), scale_valid (u8 at byte 4), then eleven zero bytes. Zero state has S=0,
scale_valid=0, and all-zero accumulators. processed_blocks remains positive.

Final output occupies three rows: 32 codes, one exponent byte, then fifteen
zero bytes. It overwrites only those three rows, after old state has been read.
The remainder of the reserved C window is not output. Invalid fields, bank
counts, source exponents, state headers, capacity or total K fail immediately.

## Compiler selection and B source layout

The isolated PolyGemvConfig compiler recognizes static M=1 products whose only
consumer is an existing last-axis MXINT8 quantization. The A, B and destination
arguments must already carry an identity buffer-layout proof. Unproven layouts,
M>1, dynamic M, FP32 fanout and resources exceeding bank capacity do not select
this specialization. The existing non-specialized mapping remains responsible
for those subgraphs; no runtime layout guess or CPU packing is introduced.

Source weights retain their existing [K-block, output-channel, packed-row]
layout. A source K block may be 512, 1024 or 1536, provided eight packed rows fit
one 16 KiB B bank. Four selected-group contiguous DMA commands load the four
sets of eight output channels. One full A row remains in its bank and serves
all N32 output tiles. N is a multiple of 32: codes are written at column offsets
and each exponent at N + column/32, never by concatenating 33-byte fragments.
Only the final source K block finalizes. The full A code/exponent row must fit
one bank, and packed bytes within a row are contiguous by the input ABI.

Compiler fixtures cover K1024 with N32/64/96 and K1536 with N32, including guard
bytes and partial exponent writes. The isolated configuration replaces one
no-GP FFN thread with a GEMV-only thread. Main Poly topology, generic matrix
mapping, dynamic prefill/decode specialization, model calibration and production
selection are not changed by this compiler gate. Whole-slice prototype timing
is recorded separately with the frozen old CPU control configuration; it must
not be mixed with later CPU result-line-buffer improvements.

The hart manager knows its thread profiles; the thread compiler only lowers its
assigned subgraph. Thread resource ratios are a separate design choice.


The model recipe must explicitly select the new MXINT8 output boundary and its
integer consumers. Any necessary FP32 conversion belongs to a declared existing
boundary. Production integration requires calibration with at most one percentage
point accuracy loss; C/BEMU agreement alone does not establish that gate.

## Affected consumers and validation

Add an isolated Toy configuration, generated ISA header, self-contained C golden
cases, BEMU implementation and registration, then bank/physical Ball operations
and small MLIR tests. Verify cross-group output order, nonzero bases, 32-channel
K512 and K1024 tiles, segmented accumulation, zero blocks, exponent jumps,
signed ties, clamp boundaries, padding and invalid inputs. Only after C/BEMU and
MLIR/BEMU gates pass may RTL, UVM, PPA and Poly integration proceed. Keep the
existing MXINT8 matmul and quantization contracts unchanged.

BEMU timing is calibrated to the ten small RTL fixtures, including last-block
zero-dot handling. Report actual RTL Ball latencies separately from whole-model
timing; calibration itself is not a measured optimization.
