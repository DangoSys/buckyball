// The cycles of one INTNORM instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle. Delays outside the Ball belong to the bank and bus model, not
// here. From IntnormBall.scala, with K values per row (rs2[15:0]), C = K / 4
// input words and K / 32 code blocks per row:
//   2      headerRead, headerWait once: the epsilon row
//   per row:
//   3 C    sum: inputRead, inputWait, sumValues per word
//   80     divide (mean): divideStep 79 down to 0
//   6 C    variance: inputRead, inputWait, multiplyStart 1 and multiplyWait 3
//          per word (Multiply(16): 2 calculate cycles, answers in the third)
//   80     divide (variance), 32 squareRoot (sqrtStep 0..31), 80 divide
//          (reciprocal)
//   20 C   affine: inputRead, inputWait, gammaRead, gammaWait, betaRead,
//          betaWait, three multiplies of 4, writing, acknowledge per word
//   21 K / 32  per block: 8 scratchRead, scratchWait pairs, encode, two
//          writing, acknowledge pairs
//   2 K / 512  per 16 blocks: one more writing, acknowledge for the exponents
//   row = 29 C + 272 + 21 K / 32 + 2 K / 512 (4322 for K = 512). The rows
//   start at cycle 3 and complete follows the last.
// No path depends on the data: the divide and square root loops run a fixed
// count.
// The UVM tests of the RTL check it exactly.

/// One INTNORM instruction as the Ball receives it: `iter` rows of rs2[15:0]
/// values.
pub fn cycles(iter: u64, rs2: u64) -> u64 {
    let k = rs2 & 65535;
    3 + iter * (29 * (k / 4) + 272 + 21 * (k / 32) + 2 * (k / 512))
}
