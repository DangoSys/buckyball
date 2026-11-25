// The cycles of one INT8ADD or INT8ADD_RELU instruction from its command
// handshake to the first cycle its response is valid, against an ideal outside
// world: every bank request ready, a read answered the next cycle, a write
// acknowledged the next cycle. Delays outside the Ball belong to the bank and
// bus model, not here. From Int8AddBall.scala:
//   2       idle -> waitForChannels -> readRequest at cycle 2
//   30 / row
//     2     readRequest (both ports fire), readResponse (both answers)
//     16    calculate, one lane per cycle; lane L issues at cycle c + L
//     10    lane 15 through the chain: IntToFloat 2, Fma 3,
//           PipelinedAddRawFN 4, roundedResult/resultValid 1; the 16th result
//           moves calculateDrain to writeRequest
//     2     writeRequest, writeResponse: one handshake each
//     (readRequest at r, calculate r + 2 .. r + 17, lane 15 resultValid at
//     r + 27, writeRequest r + 28, writeResponse r + 29, next r + 30)
//   the writeResponse that ends a group goes straight to readRequest as
//   between rows; the last writeResponse is at cycle 30 * groups * iter + 1
//   and complete presents the response the next cycle.
// RELU only masks the result; the arithmetic is a fixed pipeline with no
// data-dependent timing. The UVM tests of the RTL check it exactly.

/// One INT8ADD or INT8ADD_RELU instruction as the Ball receives it: `iter`
/// rows in each of `groups` bank groups (op1_col).
pub fn cycles(iter: u64, groups: u64) -> u64 {
    30 * groups * iter + 2
}
