package chi_link_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual chi_link_if vif;
    bit [31:0] pending[$];
    int accepted = 0, checked = 0, cycle = 0;
    bit stalled = 0;
    bit [31:0] held;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(virtual chi_link_if)::get(this, "", "vif", vif))
        `uvm_fatal("VIF", "CHI credit link missing")
    endfunction
    task monitor();
      forever begin
        @(posedge vif.clock);
        if (vif.reset) begin
          pending.delete();
          stalled = 0;
        end else begin
          cycle++;
          if (vif.valid && vif.ready) begin
            pending.push_back(vif.data);
            accepted++;
          end
          if (stalled && (!vif.out_valid || vif.out_data !== held))
            `uvm_fatal("STABLE", "Output changed while stalled")
          if (vif.out_valid && vif.out_ready) begin
            bit [31:0] golden;
            if (!pending.size()) `uvm_fatal("UNEXPECTED", "Output without request")
            golden = pending.pop_front();
            if (vif.out_data !== golden) `uvm_fatal("DATA", "Credit link data/order mismatch")
            checked++;
          end
          stalled = vif.out_valid && !vif.out_ready;
          held = vif.out_data;
          if (vif.credits > 4) `uvm_fatal("CREDIT", "Credit count exceeds depth")
        end
      end
    endtask
    task restart();
      @(negedge vif.clock);
      vif.reset  = 1;
      vif.active = 0;
      vif.valid  = 0;
      repeat (4) @(negedge vif.clock);
      vif.reset = 0;
      repeat (2) @(negedge vif.clock);
      vif.active = 1;
      repeat (8) @(negedge vif.clock);
    endtask
    task execute();
      vif.reset = 1;
      vif.active = 0;
      vif.valid = 0;
      vif.data = 0;
      vif.out_ready = 0;
      fork
        monitor();
      join_none
      restart();
      for (int i = 0; i < 256; i++) begin
        vif.valid = 1;
        vif.data  = (32'h9e3779b9 * (i + 1)) ^ ((i & 4) ? 32'hffffffff : 32'h0);
        do begin
          @(posedge vif.clock);
          if (vif.ready) begin
            @(negedge vif.clock);
            break;
          end
          @(negedge vif.clock);
          vif.out_ready = cycle % 7 < 3;
        end while (1);
        vif.out_ready = cycle % 7 < 3;
      end
      vif.valid = 0;
      vif.out_ready = 1;
      while (pending.size()) @(negedge vif.clock);
      // Coordinated reset cancels buffered traffic and restores initial credits.
      vif.out_ready = 0;
      repeat (4) begin
        vif.valid = 1;
        vif.data  = 32'hfedcba98;
        do @(posedge vif.clock); while (!vif.ready);
        @(negedge vif.clock);
      end
      vif.valid = 0;
      repeat (8) @(negedge vif.clock);
      restart();
      vif.out_ready = 1;
      for (int i = 0; i < 32; i++) begin
        vif.valid = 1;
        vif.data  = i;
        do @(posedge vif.clock); while (!vif.ready);
        @(negedge vif.clock);
      end
      vif.valid = 0;
      while (pending.size()) @(negedge vif.clock);
      repeat (8) @(negedge vif.clock);
      // A coordinated reset can interrupt a physical flit before RX captures it.
      vif.valid = 1;
      vif.data  = 32'h01234567;
      do @(posedge vif.clock); while (!vif.ready);
      @(negedge vif.clock);
      vif.reset  = 1;
      vif.active = 0;
      vif.valid  = 0;
      repeat (4) @(negedge vif.clock);
      vif.reset = 0;
      repeat (2) @(negedge vif.clock);
      vif.active = 1;
      repeat (8) @(negedge vif.clock);
      if (checked != 288 || accepted != 293 || vif.credits != 4)
        `uvm_fatal("COUNT", $sformatf(
                   "accepted=%0d checked=%0d credits=%0d", accepted, checked, vif.credits))
      `uvm_info(
          "CHECKS",
          "CHI credit link: 288 ordered outputs checked, 5 reset-cancelled (4 buffered, 1 in-flight)",
          UVM_LOW)
    endtask
  endclass
endpackage
