// The cycles of one F32ADD instruction from its command handshake to the first
// cycle its response is valid, against an ideal outside world: every bank
// request ready, a read answered the next cycle, a write acknowledged the next
// cycle and the write port taking no new write until that acknowledgement is
// accepted. Delays outside the Ball belong to the bank and bus model, not
// here. From F32AddBall.scala, with the two-entry framework Queue (enq at c,
// deq.valid at c + 1) for operands and results:
//   2       idle -> channels -> running: the first read requests at cycle 2
//   3       row 0: read answered at 3 into the operand queues, added at 4 into
//           the result queue, written at 5
//   2 / row the write port is the bottleneck: write at c, acknowledgement at
//           c + 1, the port ready again at c + 2. Reads run ahead (two credits
//           per port) and the result queue holds the next sum, so row k is
//           written at 5 + 2k and acknowledged at 6 + 2k
//   then the last acknowledgement is at cycle 2 * iter + 4 and complete
//   presents the response the next cycle.
// Both rs2[5] modes take this path: with +0 (first) port 1 is idle, port 0
// reads exactly as when accumulating and the operand-0 queue alone gates the
// add. The source group (rs2[4:0]) only offsets the address; the adder is
// combinational. The UVM tests of the RTL check it exactly.

/// One F32ADD instruction as the Ball receives it: `iter` rows.
pub fn cycles(iter: u64) -> u64 {
    2 * iter + 5
}
