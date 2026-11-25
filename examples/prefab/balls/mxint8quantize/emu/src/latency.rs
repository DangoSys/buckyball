// The cycles of one MXINT8QUANT instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle, the response accepted at once. Delays outside the Ball belong to
// the bank and bus model, not here. The timing does not depend on the data.
// From Mxint8quantizeBall.scala:
//   1        idle -> reading after the command handshake
//   per block of 32 values:
//     9      reading: eight pipelined reads issued on cycles 0..7, answered on
//            1..8; the eighth answer moves to exponent
//     1      exponent
//     4      writing: two code rows, each write acknowledged the next cycle and
//            the port free one cycle after that (fire 0, 2; last ack at 3)
//   2        a third write in writing, the scale row, at every 16th block of a
//            row and at its last block: ceil(blocks / 16) per row
//   then the last acknowledgement moves to complete, where the response is
//   valid.
// The UVM tests of the RTL check it exactly.

/// One MXINT8QUANT instruction as the Ball receives it: `iter` rows of K
/// values, K in rs2[15:0].
pub fn cycles(iter: u64, rs2: u64) -> u64 {
    let blocks = (rs2 & 0xffff) / 32;
    1 + iter * (14 * blocks + 2 * blocks.div_ceil(16))
}
