// The cycles of one MXQUANT instruction from its command handshake to the first
// cycle its response is valid, against an ideal outside world: every bank
// request ready, a read answered the next cycle, a write acknowledged the next
// cycle and ready again the cycle after. Delays outside the Ball belong to the
// bank and bus model, not here. From MxquantBall.scala, cycle 0 being the
// command handshake:
//   1       idle -> readReq: the first read request in cycle 1
//   55 / G  per 32-element block:
//             16  8 input rows, each readReq + readResp
//              1  exponent
//             32  encode, one lane per cycle (lane 0..31)
//              4  two code rows, each codeReq + codeResp
//              2  the scale byte, scaleReq + scaleResp
//   the last scaleResp is in cycle 55 * G and complete is valid the next cycle.
// No mode and no data changes the path.
// The UVM tests of the RTL check every instruction against it.

/// One MXQUANT instruction of `iter` FP32 elements (`iter / 32` blocks).
pub fn cycles(iter: u64) -> u64 {
    55 * (iter / 32) + 1
}
