`include "parameters.svh"
`define TI(F) vif.inputs[`TM_IN_``F``_OFFSET+:`TM_IN_``F``_WIDTH]
`define TO(F) vif.outputs[`TM_OUT_``F``_OFFSET+:`TM_OUT_``F``_WIDTH]
`define TB(F) vif.backend[`TM_IN_``F``_OFFSET+:`TM_IN_``F``_WIDTH]
`define TS(F) pins[`TM_IN_``F``_OFFSET+:`TM_IN_``F``_WIDTH]
`define MI(T, F) T.inputs[`TM_IN_``F``_OFFSET+:`TM_IN_``F``_WIDTH]
`define MO(T, F) T.outputs[`TM_OUT_``F``_OFFSET+:`TM_OUT_``F``_WIDTH]
package tomem_pkg;
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
endpackage
