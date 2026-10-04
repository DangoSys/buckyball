package chi_credit_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual chi_credit_if vif;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(virtual chi_credit_if)::get(this, "", "vif", vif))
        `uvm_fatal("VIF", "Credit interface missing")
    endfunction
    task execute();
      vif.reset  = 1;
      vif.active = 0;
`ifdef CHI_TX_OVERFLOW
      vif.lcrdv = 0;
`else
      vif.pend  = 0;
      vif.valid = 0;
      vif.flit  = 0;
`endif
      repeat (4) @(negedge vif.clock);
      vif.reset  = 0;
      vif.active = 1;
`ifdef CHI_TX_OVERFLOW
      // Peer returns five credits to a four-credit transmitter with no traffic.
      vif.lcrdv = 1;
      repeat (8) @(negedge vif.clock);
`else
      // Peer uses a fifth flit after exhausting four credits; receiver cannot drain.
      vif.pend = 1;
      repeat (8) @(negedge vif.clock);
      vif.valid = 1;
      repeat (6) @(negedge vif.clock);
`endif
      `uvm_fatal("NEGATIVE", "Credit violation did not terminate RTL")
    endtask
  endclass
endpackage
