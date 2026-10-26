`include "parameters.svh"
`define BI(P, N) P[`BAT_IN_``N``_OFFSET +: `BAT_IN_``N``_WIDTH]
`define BO(P, N) P[`BAT_OUT_``N``_OFFSET +: `BAT_OUT_``N``_WIDTH]
package bat_pkg;
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
