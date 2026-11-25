// The cycles of one INT8MUL instruction from its command handshake to the first
// cycle its response is valid, against an ideal outside world: every bank
// request ready, a read answered the next cycle, a write acknowledged the next
// cycle. Delays outside the Ball belong to the bank and bus model, not here.
// From Int8MulBall.scala:
//   2       idle -> waitForChannels -> gateRequest at cycle 2
//   2 / group gateRequest, gateResponse: the gate row, read again for every
//           group, then inputRequest
//   26 / row
//     2     inputRequest, inputResponse: one handshake each
//     16    calculate, one lane per cycle; lane L issues at cycle c + L
//     6     lane 15 through the scale chain: productReg (+1), productFloatReg
//           and floatValid (+2), Fma latency 3 (+5), scaledValid (+6); in
//           calculateDrain scaledValid for lane 15 moves to writeRequest
//     2     writeRequest, writeResponse: one handshake each
//     (inputRequest at c, lane 15 at c + 17, its scaledValid at c + 23,
//     writeRequest c + 24, writeResponse c + 25, next inputRequest c + 26)
//   a group is 26 * iter + 2 cycles from its gateRequest to the next one; the
//   last writeResponse is at 2 + groups * (26 * iter + 2) - 1 and complete
//   presents the response the next cycle.
// The arithmetic has no data-dependent timing. The UVM tests of the RTL check
// it exactly.

/// One INT8MUL instruction as the Ball receives it: `iter` rows in each of
/// `groups` bank groups (op1_col).
pub fn cycles(iter: u64, groups: u64) -> u64 {
    groups * (26 * iter + 2) + 2
}
