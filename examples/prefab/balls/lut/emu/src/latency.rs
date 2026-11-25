// The cycles of one LUT instruction from its command handshake to the first
// cycle its response is valid, against an ideal outside world: every bank
// request ready, a read answered the next cycle, a write acknowledged the next
// cycle. Delays outside the Ball belong to the bank and bus model, not here.
// From LutBall.scala every bank state takes one handshake; the first inputReq
// is at cycle 1 and complete follows the last state. For R = iter rows:
//   mode 0 (byte lookup; a lane table, op2_col = 4, only changes the address):
//     36 / row   inputReq, inputResp, 16 * (lutReq, lutResp), writeReq,
//                writeResp
//   mode 1 (word lookup):
//     12 / row   inputReq, inputResp, 4 * (lutReq, lutResp), writeReq,
//                writeResp
//   mode 2 (softmax):
//     2 / row    phase 0 (maximum): inputReq, inputResp
//     12 / row   phase 1 (table): as mode 1; the last writeResp -> divideReq
//     66         divideReq (DivRem(64) accepts), divideResp: DivRem iterates
//                64 cycles (sew 3; the sum holds Q30 one, never zero) and
//                presents the quotient the cycle after, accepted at once
//     26 / row   phase 2 (normalize): inputReq, inputResp, 4 * (multiplyReq,
//                multiplyResp): Multiply(32) accepts, 4 radix-256 steps,
//                complete the cycle after, accepted at once: 6 per lane
//     2 / write  phase 2 writes after every fourth row and the last:
//                ceil(R / 4); phase 3 (cleanup) writes rows ceil(R / 4) ..
//                R - 1: R - ceil(R / 4). Together 2 * R.
//     (2 + 12 + 26 + 2) * R + 66
//   complete   the response is valid the cycle after the last state (+1).
// The UVM tests of the RTL check it exactly.

/// One LUT as the Ball receives it: `iter` rows in the mode rs2[1:0].
pub fn cycles(iter: u64, rs2: u64) -> u64 {
    match rs2 & 3 {
        0 => 36 * iter + 1,
        1 => 12 * iter + 1,
        2 => 42 * iter + 66 + 1,
        mode => panic!("lut: mode {mode} has no timing"),
    }
}
