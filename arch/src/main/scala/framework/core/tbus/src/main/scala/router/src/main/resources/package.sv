`include "parameters.svh"
`define RI(T, F) T.inputs[`TR_IN_``F``_OFFSET+:`TR_IN_``F``_WIDTH]
`define RO(T, F) T.outputs[`TR_OUT_``F``_OFFSET+:`TR_OUT_``F``_WIDTH]
package tbus_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  `include "agent/item.svh"
  `include "agent/fixture.svh"
  `include "agent/driver.svh"
  `include "agent/monitor.svh"
  `include "agent/peer.svh"
  `include "env/model.svh"
  `include "env/env.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
  `include "tests/invalid.svh"
endpackage
