`include "parameters.svh"
`define AI(T, F) T.inputs[`AD_IN_``F``_OFFSET+:`AD_IN_``F``_WIDTH]
`define AO(T, F) T.outputs[`AD_OUT_``F``_OFFSET+:`AD_OUT_``F``_WIDTH]
`define EXPECT(F,
               V) want[`AD_OUT_``F``_OFFSET+:`AD_OUT_``F``_WIDTH]=V;mask[`AD_OUT_``F``_OFFSET+:`AD_OUT_``F``_WIDTH]='1;
package admission_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  `include "../common/agent/item.svh"
  `include "../common/agent/driver.svh"
  `include "../common/agent/monitor.svh"
  `include "env/model.svh"
  `include "env/env.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
endpackage
