// The cycles of one INTSOFTMAX instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle. Delays outside the Ball belong to the bank and bus model, not
// here. From IntSoftmaxBall.scala, with K scores per row (rs2[7:0]) and
// S = ceil(K / 4) words per row, a row costs:
//   S + 1  loadScores: S read requests back to back, each answered the next
//          cycle; the last response moves on
//   2 K    lookupRequest, lookupResponse per score: one table read each
//   66     divideRequest 1, then DivRem(64) runs 64 iterate cycles and answers
//          from complete: divideResponse 65
//   12 K   per score multiplyQuotient 1 + quotientResponse 5, multiplyProduct
//          1 + productResponse 5: Multiply(32) takes 4 calculate cycles and
//          answers in the fifth
//   2 S    writeRequest, writeResponse per word of four results
//   row = 14 K + 3 S + 67. The command handshake is cycle 0, loadScores starts
//   at cycle 1 and complete follows the last row.
// No path depends on the data: the table index only selects an address, and
// the divisor (the sum of the weights) is never zero.
// The UVM tests of the RTL check it exactly.

/// One INTSOFTMAX instruction as the Ball receives it: `iter` rows of
/// rs2[7:0] scores.
pub fn cycles(iter: u64, rs2: u64) -> u64 {
    let k = rs2 & 255;
    let s = k.div_ceil(4);
    1 + iter * (14 * k + 3 * s + 67)
}
