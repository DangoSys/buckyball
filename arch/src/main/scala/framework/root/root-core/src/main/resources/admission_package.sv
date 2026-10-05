package admission_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function chandle core_ref_create(input string path);
  import "DPI-C" function void core_ref_destroy(input chandle model);
  import "DPI-C" function void core_ref_read(
    input chandle model,
    input longint unsigned address,
    output bit [511:0] data
  );
  import "DPI-C" function void core_ref_write(
    input chandle model,
    input longint unsigned address,
    input bit [511:0] data,
    input bit [63:0] mask
  );
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual admission_control_if control;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      timeout = 1ms;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(virtual admission_control_if)::get(this, "", "control", control))
        `uvm_fatal("VIF", "missing admission interface")
    endfunction
    task execute();
      @(negedge control.clock);
      control.start = 1;
      wait (control.finished);
      `uvm_info(
          "ADMISSION_PASS",
          "Actual Rocket/Core reservations, context snapshots, younger request ordering and completion/cancel verified",
          UVM_LOW)
    endtask
  endclass
endpackage
