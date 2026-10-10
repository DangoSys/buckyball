package rvv_pkg;
  timeunit 1ns; timeprecision 1ps;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function void rvv_model_init();
  import "DPI-C" function int unsigned rvv_case_count();
  import "DPI-C" function void rvv_case_select(int unsigned n);
  import "DPI-C" function longint unsigned rvv_case_meta(int unsigned field);
  import "DPI-C" function int unsigned rvv_image_bytes();
  import "DPI-C" function int unsigned rvv_program_word(int unsigned n);
  import "DPI-C" function int unsigned rvv_image_word(int unsigned n);
  import "DPI-C" function void rvv_memory_masked_write(
    longint unsigned address,
    longint unsigned data,
    int unsigned mask
  );
  import "DPI-C" function longint unsigned rvv_expected_status(int unsigned field);
  import "DPI-C" function longint unsigned rvv_memory_read(
    longint unsigned address,
    int unsigned size
  );
  import "DPI-C" function int unsigned rvv_memory_error(
    longint unsigned address,
    int unsigned size
  );
  import "DPI-C" function void rvv_memory_write(
    longint unsigned address,
    int unsigned size,
    longint unsigned data,
    int unsigned mask
  );
  import "DPI-C" function int unsigned rvv_compare_memory();
  import "DPI-C" function int unsigned rvv_compare_image();

  class launch_item extends uvm_sequence_item;
    bit execute;
    bit protocol_fault;
    bit [31:0] protocol_cause, protocol_pc, protocol_instruction;
    bit [63:0] protocol_tval;
    bit [3:0] rob;
    `uvm_object_utils(launch_item)
    function new(string name = "launch_item");
      super.new(name);
    endfunction
  endclass
  class completion_item extends uvm_sequence_item;
    bit fault;
    bit [15:0] write_bank;
    bit [3:0] rob;
    bit [31:0] pc, instruction, cause;
    bit [63:0] tval;
    `uvm_object_utils_begin(completion_item)
      `uvm_field_int(fault, UVM_DEFAULT)
      `uvm_field_int(rob, UVM_DEFAULT)
      `uvm_field_int(write_bank, UVM_DEFAULT)
      `uvm_field_int(pc, UVM_DEFAULT)
      `uvm_field_int(instruction, UVM_DEFAULT | UVM_NOCOMPARE)
      `uvm_field_int(cause, UVM_DEFAULT)
      `uvm_field_int(tval, UVM_DEFAULT)
    `uvm_object_utils_end
    function new(string name = "completion_item");
      super.new(name);
    endfunction
    function bit do_compare(uvm_object rhs, uvm_comparer comparer);
      completion_item other;
      if (!$cast(other, rhs)) return 0;
      return super.do_compare(rhs, comparer) && (!fault || instruction == other.instruction);
    endfunction
  endclass
  class kernel_model extends reference_model #(launch_item, completion_item);
    `uvm_component_utils(kernel_model)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void write(launch_item t);
      completion_item expected = completion_item::type_id::create("expected");
      expected.rob = t.rob;
      expected.fault = t.execute ? rvv_expected_status(0) : 0;
      expected.pc = t.execute ? rvv_expected_status(1) : 0;
      expected.instruction = t.execute ? rvv_expected_status(2) : 0;
      expected.cause = t.execute ? rvv_expected_status(3) : 0;
      expected.tval = t.execute ? rvv_expected_status(4) : 0;
      if (t.protocol_fault) begin
        expected.fault = 1;
        expected.pc = t.protocol_pc;
        expected.instruction = t.protocol_instruction;
        expected.cause = t.protocol_cause;
        expected.tval = t.protocol_tval;
      end
      expected.write_bank = t.execute && !expected.fault ? 1 : 0;
      expected_ap.write(expected);
    endfunction
  endclass
  typedef checked_env#(launch_item, completion_item, kernel_model) kernel_env;
  `include "test.svh"
endpackage
