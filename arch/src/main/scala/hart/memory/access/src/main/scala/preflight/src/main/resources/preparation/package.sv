`include "preparation_parameters.svh"
`define PI(F) vif.inputs[`PREP_IN_``F``_OFFSET+:`PREP_IN_``F``_WIDTH]
`define PO(F) vif.outputs[`PREP_OUT_``F``_OFFSET+:`PREP_OUT_``F``_WIDTH]
`define PB(F) vif.backend[`PREP_IN_``F``_OFFSET+:`PREP_IN_``F``_WIDTH]
`define PS(F) pins[`PREP_IN_``F``_OFFSET+:`PREP_IN_``F``_WIDTH]
`define AI(T, F) T.inputs[`PREP_IN_``F``_OFFSET+:`PREP_IN_``F``_WIDTH]
`define AO(T, F) T.outputs[`PREP_OUT_``F``_OFFSET+:`PREP_OUT_``F``_WIDTH]
package preparation_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function chandle preflight_ref_create(
    input int unsigned address_bits,
    beat
  );
  import "DPI-C" function void preflight_ref_destroy(input chandle model);
  import "DPI-C" function void preflight_ref_program(
    input chandle model,
    input longint unsigned address,
    value,
    input int unsigned error
  );
  import "DPI-C" function void preflight_ref_policy(
    input chandle model,
    input int unsigned pte_valid,
    input longint unsigned pte,
    input int unsigned range_valid,
    input longint unsigned first,
    last
  );
  import "DPI-C" function int unsigned preflight_ref_submit(
    input chandle model,
    input int unsigned id,
    input longint unsigned base,
    input int unsigned rows,
    cols,
    span,
    col_stride,
    row_stride,
    write,
    mode,
    input longint unsigned root,
    input int unsigned privilege,
    sum,
    mxr
  );
  import "DPI-C" function int unsigned preflight_ref_count(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void preflight_ref_expected(
    input chandle model,
    input int unsigned id,
    index,
    output longint unsigned va,
    pa,
    output int unsigned bytes,
    write,
    last,
    error
  );
  import "DPI-C" function int unsigned preflight_ref_authorize(
    input chandle model,
    input int unsigned id,
    input longint unsigned pa,
    input int unsigned bytes,
    write,
    pte,
    privilege,
    output int unsigned allow
  );
  import "DPI-C" function int unsigned preflight_ref_pte(
    input chandle model,
    input int unsigned id,
    input longint unsigned address,
    output longint unsigned value,
    output int unsigned error
  );
  import "DPI-C" function int unsigned preflight_ref_pending(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void preflight_ref_retire(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void preflight_ref_reset(input chandle model);
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
