// The cycles of one MXINT8UNPACK instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle, the response accepted at once. Delays outside the Ball belong to
// the bank and bus model, not here. The timing does not depend on the data.
// From Mxint8UnpackBall.scala:
//   1        idle -> metadataRequest after the command handshake
//   2        metadataRequest + metadataResponse, before blocks 0, 16, 32, ...:
//            ceil(blocks / 16) times
//   per block:
//     3      codes: two pipelined reads issued on cycles 0, 1, answered on
//            1, 2; the second answer moves to write
//     16     write: eight rows, each acknowledged the next cycle and the port
//            free one cycle after that (fire 0, 2, ..., 14; last ack at 15)
//   then the last acknowledgement moves to complete, where the response is
//   valid.
// The UVM tests of the RTL check it exactly.

/// One MXINT8UNPACK instruction as the Ball receives it: `iter` blocks.
pub fn cycles(iter: u64) -> u64 {
    1 + 2 * iter.div_ceil(16) + 19 * iter
}
