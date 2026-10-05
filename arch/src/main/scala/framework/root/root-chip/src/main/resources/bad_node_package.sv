package chi_bad_node_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual chi_codec_if vif;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(virtual chi_codec_if)::get(this, "", "vif", vif))
        `uvm_fatal("VIF", "codec missing")
    endfunction
    task execute();
      vif.reset = 1;
      vif.pause = 0;
      vif.valid = 0;
      vif.target = 0;
      vif.data = 0;
      vif.out_ready = 1;
      repeat (4) @(negedge vif.clock);
      vif.reset = 0;
      vif.valid = 1;
      repeat (8) @(negedge vif.clock);
      `uvm_fatal("MISSING_ASSERTION", "unplaced NodeID was not rejected")
    endtask
  endclass
endpackage
