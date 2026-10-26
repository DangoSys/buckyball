`include "parameters.svh"
`define GI(P, N) P[`GROB_IN_``N``_OFFSET+:`GROB_IN_``N``_WIDTH]
`define GO(P, N) P[`GROB_OUT_``N``_OFFSET+:`GROB_OUT_``N``_WIDTH]
`define OP(P,
           N) P[`GROB_IN_ALLOC_BITS_OPERATION_``N``_OFFSET-`GROB_IN_OPERATION_OFFSET+:`GROB_IN_ALLOC_BITS_OPERATION_``N``_WIDTH]
`define CP(
    P, N) P[`GROB_IN_ALLOC_BITS_``N``_OFFSET-`GROB_IN_ALLOC_OFFSET+:`GROB_IN_ALLOC_BITS_``N``_WIDTH]
`define BA(P,
           N) P[`GROB_IN_BAT_RENAMED_``N``_OFFSET-`GROB_IN_RENAMED_OFFSET+:`GROB_IN_BAT_RENAMED_``N``_WIDTH]
`define ISS(
    P, N) P[`GROB_OUT_ISSUE_BITS_``N``_OFFSET-`GROB_ISSUE_OFFSET+:`GROB_OUT_ISSUE_BITS_``N``_WIDTH]
package global_rob_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  `include "agent/item.svh"
  `include "env/model.svh"
  `include "agent/driver.svh"
  `include "agent/monitor.svh"
  `include "agent/peer.svh"
  `include "env/env.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
endpackage
