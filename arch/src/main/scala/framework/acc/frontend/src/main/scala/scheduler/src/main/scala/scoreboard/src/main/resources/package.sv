`include "parameters.svh"
`define BI(P, N) P[`SB_IN_``N``_OFFSET +: `SB_IN_``N``_WIDTH]
`define BO(P, N) P[`SB_OUT_``N``_OFFSET +: `SB_OUT_``N``_WIDTH]
package scoreboard_pkg;
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
