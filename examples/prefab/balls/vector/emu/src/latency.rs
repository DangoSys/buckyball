// The cycles of one VECMAT16 instruction from its command handshake to the
// first cycle its response is valid, against an ideal outside world: every
// bank request ready, a read answered the next cycle, a write acknowledged the
// next cycle. Delays outside the Ball belong to the bank and bus model, not
// here. From VecMat16.scala, VecMat16AccLoad.scala and VecMat16Store.scala,
// with K = iter rows of A^T and B:
//   1      sIdle starts the AccLoad at the handshake; AccLoad is in sReq at
//          cycle 1
//   64     AccLoad: 16 rows x 2 phases of sReq (both reads), sResp (both
//          answers)
//   1      AccLoad sFinish raises done; sReadReq starts at cycle 66
//   2 K    sReadReq (A and B requests), sReadResp (both answers) per row
//   32     Store: 16 rows of sReq (four writes), sResp (four
//          acknowledgements); done in the last sResp
//   1      sDone holds the response
// No path depends on the data: the instruction always accumulates.
// The UVM tests of the RTL check it exactly.

/// One VECMAT16 instruction as the Ball receives it: `iter` rows of A^T and B.
pub fn cycles(iter: u64) -> u64 {
    98 + 2 * iter
}
