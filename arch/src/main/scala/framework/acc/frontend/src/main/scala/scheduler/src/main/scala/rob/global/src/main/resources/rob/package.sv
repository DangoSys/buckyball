`include "parameters.svh"
`define RI(P, N) P[`ROB_IN_``N``_OFFSET+:`ROB_IN_``N``_WIDTH]
`define RO(P, N) P[`ROB_OUT_``N``_OFFSET+:`ROB_OUT_``N``_WIDTH]
package rob_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  `include "agent/item.svh"
  `include "agent/driver.svh"
  `include "agent/monitor.svh"
  `include "env/model.svh"
  `include "env/env.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
endpackage
