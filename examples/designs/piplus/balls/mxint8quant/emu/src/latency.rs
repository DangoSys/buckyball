// The cycles of one MXINT8QUANT instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle (the port free the cycle after), the response accepted at once.
// From Mxint8quantBall.scala: the input is n groups of G bank rows, group g
// read at cycles 1 + Gg .. G + Gg, one bank row a cycle. Its last answer sets
// full(slot), so it is written from W = G + 2 + Gg: C rows, one write every 2
// cycles (accept, acknowledge), the last at W + 2(C - 1). That is inside the
// G cycles of the next group's reads (at most 2 * 3 < 8, 2 * 12 < 32), so the
// reads set the pace: group g + 2 waits for drained > g, at W + 2C - 1, no
// later than its first read at W + G - 1. The last write's acknowledge clears
// full, outstanding is 0 the cycle after, the done condition holds there and
// done raises the response the next: W + 2C + 1 for the last group, whose
// block is its rows' last, so it writes the exponent rows too:
//   rows:    G = 8, a block of a row; C = 2 + 1
//   columns: G = 32, a block of four rows (rs2 bit 44); C = 4 * 2 + 4
// G * n + 2C + 3 in all.
// The UVM tests of the RTL check it exactly.

/// One MXINT8QUANT instruction as the Ball receives it: `iter` rows of K.
pub fn cycles(iter: u64, rs2: u64) -> u64 {
    let k = rs2 & 0xffff;
    let (g, c) = if (rs2 >> 44) & 1 == 1 { (32, 12) } else { (8, 3) };
    let n = iter * (k / 4) / g;
    g * n + 2 * c + 3
}
