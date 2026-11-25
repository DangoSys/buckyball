// The cycles of one QUANT_F32_TO_I8 or QUANT_I32_TO_I8 instruction from its
// command handshake to the first cycle its response is valid, against an
// ideal outside world: every bank request ready, a read answered the next
// cycle, a write acknowledged the next cycle. Delays outside the Ball belong to
// the bank and bus model, not here. Both share one datapath in
// ToInt8Ball.scala (the I32 INToRecFN is combinational; rs2 only places the
// output rows), cycle 0 being the command handshake:
//   1        idle -> scaleReq
//   8 / group scaleReq, scaleResp for each of its four scale rows
//   13 / row inputReq, inputResp, pack 4 (lanes 0..3 issued one a cycle);
//            lane k issued at p passes inputStage0 (p+1), multiplyInput
//            (p+2, Fma validin), Fma(3) validout (p+5), multiplyResult (p+6),
//            quantizedResult (valid p+7): lane 3 is done 7 cycles after pack
//            and packDrain moves on
//   2 / 4 rows every fourth row writes its packed word: writeReq, writeResp
//   the next group restarts at scaleReq; complete follows the last writeResp.
// The UVM tests of the RTL check it on every instruction.

/// One quantize instruction as the Ball receives it: `iter` input rows in each
/// of its `cols` channel groups.
pub fn cycles(iter: u64, cols: u64) -> u64 {
    1 + cols * (8 + 13 * iter + 2 * (iter / 4))
}
