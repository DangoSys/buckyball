package instruction_pkg;
  timeunit 1ns; timeprecision 1ps;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function void instruction_expected(
    input int unsigned bits,
    output int unsigned expanded,
    output byte unsigned illegal,
    output longint unsigned immediate,
    output int unsigned atomic,
    function_code,
    output byte unsigned word,
    immediate_mode
  );
  `include "agent/item.svh"
  `include "agent/driver.svh"
  `include "agent/monitor.svh"
  `include "agent/agent.svh"
  `include "env/model.svh"
  `include "env/coverage.svh"
  `include "env/env.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
endpackage
