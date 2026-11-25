// The cycles of one MATADD instruction from its command handshake to the first
// cycle its response is valid, against an ideal outside world: every bank
// request ready, a read answered the next cycle, a write acknowledged the next
// cycle. Delays outside the Ball belong to the bank and bus model, not here.
// From MatAdd.scala:
//   2       idle -> waitForChannels -> readRequest: the first read requests
//   4 / row readRequest (both ports fire), readResponse (both answers),
//           writeRequest, writeResponse, one handshake each, read and write
//           answered the next cycle; the writeResponse that ends a group goes
//           straight to readRequest as between rows
//   then the last writeResponse is at cycle 4 * groups * iter + 1 and complete
//   presents the response the next cycle.
// The arithmetic has no data-dependent timing. The UVM tests of the RTL check
// it exactly.

/// One MATADD instruction as the Ball receives it: `iter` rows in each of
/// `groups` bank groups (op1_col).
pub fn cycles(iter: u64, groups: u64) -> u64 {
    4 * groups * iter + 2
}
