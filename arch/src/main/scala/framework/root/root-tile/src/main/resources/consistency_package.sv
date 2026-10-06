package consistency_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function chandle cons_ref_create();
  import "DPI-C" function void cons_ref_destroy(input chandle model);
  import "DPI-C" function longint unsigned cons_ref_expected64(
    input chandle model,
    input longint unsigned address
  );
  import "DPI-C" function void cons_ref_cpu_commit(
    input chandle model,
    input longint unsigned address,
    data,
    input int unsigned mask
  );
  import "DPI-C" function void cons_ref_ddr_read(
    input chandle model,
    input longint unsigned address,
    output bit [511:0] data
  );
  import "DPI-C" function int unsigned cons_ref_write_matches(
    input chandle model,
    input longint unsigned address,
    input bit [511:0] data,
    input bit [63:0] mask
  );
  import "DPI-C" function void cons_ref_ddr_commit(
    input chandle model,
    input longint unsigned address,
    input bit [511:0] data,
    input bit [63:0] mask
  );
  import "DPI-C" function int unsigned cons_ref_ddr_visible(
    input chandle model,
    input longint unsigned address
  );
  import "DPI-C" function void cons_ref_dma_write(
    input chandle model,
    input longint unsigned address,
    data,
    input int unsigned mask
  );
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual consistency_control_if control;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      timeout = 1ms;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(virtual consistency_control_if)::get(this, "", "control", control))
        `uvm_fatal("VIF", "missing consistency control")
    endfunction
    task execute();
      @(negedge control.clock);
      control.start = 1;
      wait (control.finished);
      `uvm_info(
          "CONSISTENCY_PASS",
          "Real L1/CMO/Home DDR data, partial DMA bytes, delayed ACK/old fill and CPU permit checks completed",
          UVM_LOW)
    endtask
  endclass
endpackage
