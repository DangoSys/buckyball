// The cycles of one BDB_COUNTER instruction from its command handshake to the
// first cycle its response is valid. From Trace.scala:
//   1      idle -> sCounter: START, STOP and READ all act in the one sCounter
//          cycle
//   1      sCounter -> complete, which holds the response
// The instruction touches no bank and its subcommand does not change the path.
// The UVM tests of the RTL check it exactly.

/// One BDB_COUNTER instruction.
pub fn cycles() -> u64 {
    2
}
