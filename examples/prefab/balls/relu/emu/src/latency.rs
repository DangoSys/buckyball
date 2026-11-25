// The cycles of one RELU instruction from its command handshake to its
// response handshake against an ideal outside world: every bank request ready,
// a read answered the next cycle, a write acknowledged the next cycle, the
// response accepted at once. Delays outside the Ball belong to the bank and bus
// model, not here. From Relu.scala:
//   2       idle -> waitForChannels -> readRequest: the first read request
//   4 / row readRequest, readResponse, writeRequest, writeResponse, one
//           handshake each, read and write answered the next cycle; a new
//           segment starts like a new row
//   then the last writeResponse is at cycle 4 * rows + 1 and the response,
//   held in complete, is accepted the next cycle.
// The UVM perf tests of the RTL check it exactly.

/// One RELU instruction as the Ball receives it: `iter` rows in each of the
/// `bank_entries / rs2` segments of its bank (rs2 is the stride).
pub fn cycles(iter: u64, rs2: u64, bank_entries: u64) -> u64 {
    4 * (bank_entries / rs2 * iter) + 2
}
