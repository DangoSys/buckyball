// The cycles of one MXINT8ACT instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle, the response accepted at once. Delays outside the Ball belong to
// the bank and bus model, not here. Each lane's cost depends on its code and
// its block exponent, so the model reads them from the input bank. From
// Mxint8actBall.scala, one cycle per state:
//   1        idle -> inputReq after the command handshake
//   per block of 32 values:
//     6      inputReq + inputWait for code row 0, code row 1, the scale row
//     6      gated only: the same three reads of the gate half
//     per lane, by the path its code and exponent take in startLane:
//       2    tail (exponent < -24, or |x| at or beyond the table range):
//            startLane, commit
//       7    interior: startLane, tableReq, tableWait, multiply, interpolate,
//            rescale, commit
//       9    next (interior with the entry in word 3 of its table row, not
//            entry 4095): interior + tableNextReq, tableNextWait for the
//            neighbour in the next row
//       gating does not change a lane: commit multiplies in the same cycle
//     1      selectScale
//     32     encode, one lane per cycle
//     4      writeReq + writeWait for the two code rows
//   2        writeReq + writeWait of the scale row, at every 16th block of a
//            row and at its last block
//   then the last acknowledgement moves to complete, where the response is
//   valid.
// The UVM tests of the RTL check it exactly, lane by lane.

/// One MXINT8ACT instruction as the Ball receives it: `iter` rows, rs2 with
/// gate (bit 0), K (16:1) and the input base row (30:17); `input` is the input
/// bank, row r at bytes 16 r.
pub fn cycles(iter: u64, rs2: u64, input: &[u8]) -> u64 {
    let gated = rs2 & 1 != 0;
    let k = ((rs2 >> 1) & 0xffff) as usize;
    let base = ((rs2 >> 17) & 0x3fff) as usize * 16;
    let stride = (k / 16 + k.div_ceil(512)) * 16;
    let blocks = k / 32;
    let mut cycles = 1;
    for row in 0..iter as usize {
        let source = base + row * stride;
        for block in 0..blocks {
            cycles += 6 + if gated { 6 } else { 0 } + 1 + 32 + 4;
            let exponent = i32::from(input[source + k + block]) - 127;
            for lane in 0..32 {
                let code = i64::from(input[source + block * 32 + lane] as i8);
                if exponent < -24 {
                    cycles += 2;
                    continue;
                }
                let x = code << (exponent.min(4) + 26);
                let index = (x + (16 << 26)) >> 19;
                cycles += if x >= 16 << 26 || x <= -(16 << 26) {
                    2
                } else if index & 3 == 3 && index != 4095 {
                    9
                } else {
                    7
                };
            }
            if block % 16 == 15 || block + 1 == blocks {
                cycles += 2;
            }
        }
    }
    cycles
}
