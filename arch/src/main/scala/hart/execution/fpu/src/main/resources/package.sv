package fpu_pkg;
  timeunit 1ns; timeprecision 1ps;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function int unsigned fpu_case_count();
  import "DPI-C" function int unsigned fpu_cancel_case();
  import "DPI-C" function void fpu_case(
    input int unsigned index,
    output int unsigned instruction,
    output longint unsigned integer_value,
    a,
    rhs,
    c,
    output int unsigned rounding
  );
  import "DPI-C" function void fpu_expected(
    input int unsigned instruction,
    input longint unsigned integer_value,
    a,
    rhs,
    c,
    input int unsigned rounding,
    output longint unsigned value,
    output int unsigned flags,
    output byte unsigned integer_result,
    illegal
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
