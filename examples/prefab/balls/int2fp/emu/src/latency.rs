// The cycles of one INT32_TO_FP32 instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every bank
// request ready, a read answered the next cycle, a write acknowledged the next
// cycle. Delays outside the Ball belong to the bank and bus model, not here.
// From Int2Fp.scala, cycle 0 being the command handshake:
//   1       idle -> readReq
//   9 / row readReq (both ports: input row, scale row), readResp (both),
//           convert (INT32 -> FP32), multiply 4 (Fma(3): validin its first
//           cycle, validout three cycles later), writeReq, writeResp; the
//           next channel group starts like the next row
//   then complete follows the last writeResp.
// The UVM tests of the RTL check it on every instruction.

/// One INT32_TO_FP32 instruction as the Ball receives it: `iter` rows in each
/// of its `cols` channel groups.
pub fn cycles(iter: u64, cols: u64) -> u64 {
    1 + 9 * iter * cols
}
