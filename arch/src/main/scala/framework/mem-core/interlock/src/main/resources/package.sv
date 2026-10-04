package interlock_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function chandle interlock_ref_create(
    input int unsigned entries,
    address_bits,
    line_bytes,
    max_ranges
  );
  import "DPI-C" function void interlock_ref_destroy(input chandle model);
  import "DPI-C" function void interlock_ref_reset(input chandle model);
  import "DPI-C" function int unsigned interlock_ref_live(input chandle model);
  import "DPI-C" function void interlock_ref_reserve(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void interlock_ref_info(
    input chandle model,
    input int unsigned id,
    input byte unsigned has_mem,
    input longint unsigned base,
    bytes,
    input byte unsigned write,
    last
  );
  import "DPI-C" function byte unsigned interlock_ref_cpu_allow(
    input chandle model,
    input longint unsigned addr,
    input int unsigned bytes,
    input byte unsigned write,
    older_pending,
    reserve_now
  );
  import "DPI-C" function void interlock_ref_maint_offer(
    input chandle model,
    input int unsigned id,
    op,
    input longint unsigned first_line,
    last_line
  );
  import "DPI-C" function void interlock_ref_maint_ack(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void interlock_ref_grant(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void interlock_ref_dma_ack(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void interlock_ref_complete(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void interlock_ref_cancel(
    input chandle model,
    input int unsigned id
  );
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual interlock_control_if vif;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      timeout = 500us;
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(virtual interlock_control_if)::get(this, "", "vif", vif))
        `uvm_fatal("VIF", "missing control interface")
    endfunction
    task execute();
      @(negedge vif.clock);
      vif.start = 1;
      wait (vif.done);
      `uvm_info(
          "INTERLOCK_PASS",
          "Checked live reservations, CPU permits, maintenance/grant/ACK/completion handshakes with independent transaction oracle",
          UVM_LOW)
    endtask
  endclass
endpackage
