// The cycles of one MXFP2INT instruction (MXFP4, one input word per block) from
// its command handshake to the first cycle its response is valid, against an
// ideal outside world: every bank request ready, a read answered the next
// cycle, a write acknowledged the next cycle and the port free again the cycle
// after. Delays outside the Ball belong to the bank and bus model, not here.
// From Mxfp2Int.scala, cycle 0 being the command handshake:
//   1         idle -> sReadReq
//   8 / block sReadReq, sReadResp (port 0), sScaleReq, sScaleResp (port 1),
//             sWriteLoReq, sWriteLoResp, sWriteHiReq, sWriteHiResp, one
//             handshake each
//   then sComplete follows the last sWriteHiResp.
// The UVM tests of the RTL check it on every instruction.

/// One MXFP2INT instruction as the Ball receives it: `iter` MXFP4 blocks.
pub fn cycles(iter: u64) -> u64 {
    1 + 8 * iter
}
