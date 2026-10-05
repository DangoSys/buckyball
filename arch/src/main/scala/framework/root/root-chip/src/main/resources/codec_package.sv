package chi_codec_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual chi_codec_if vif;
    bit [388:0] expected[$];
    bit [6:0] targets[$];
    int checked = 0;
    int cycles = 0;
    bit flow = 0;
    bit stalled;
    bit [388:0] held;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(virtual chi_codec_if)::get(this, "", "vif", vif))
        `uvm_fatal("VIF", "codec missing")
    endfunction
    task monitor();
      forever begin
        @(posedge vif.clock);
        if (vif.reset) begin
          expected.delete();
          targets.delete();
          stalled = 0;
        end else begin
          if (vif.valid && vif.ready) begin
            expected.push_back(vif.data);
            targets.push_back(vif.target);
          end
          if (vif.chunk_fire) begin
            bit [6:0] target;
            if (!targets.size()) `uvm_fatal("NODEMAP_EXTRA", "chunk without accepted target")
            target = targets[0];
            if (vif.chunk_x !== (target & 1) || vif.chunk_y !== ((target >> 1) & 1))
              `uvm_fatal("NODEMAP", "CHI NodeID coordinate lookup mismatch")
            if (vif.chunk_tail) target = targets.pop_front();
          end
          if (stalled && (!vif.out_valid || vif.out_data !== held))
            `uvm_fatal("STABLE", "reassembled flit changed under stall")
          if (vif.out_valid && vif.out_ready) begin
            bit [388:0] golden;
            if (!expected.size()) `uvm_fatal("EXTRA", "unexpected reassembled CHI DAT")
            golden = expected.pop_front();
            if (vif.out_data !== golden)
              `uvm_fatal("DATA", "CHI DAT bit packing/reassembly mismatch");
            checked++;
          end
          stalled = vif.out_valid && !vif.out_ready;
          held = vif.out_data;
        end
      end
    endtask
    task flow_control();
      forever begin
        @(negedge vif.clock);
        cycles++;
        if (flow) begin
          vif.pause = cycles % 7 < 2;
          vif.out_ready = cycles % 11 >= 4;
        end else begin
          vif.pause = 0;
          vif.out_ready = 1;
        end
      end
    endtask
    task execute();
      vif.reset = 1;
      vif.pause = 0;
      vif.valid = 1;
      vif.target = 0;
      vif.data = 0;
      vif.out_ready = 0;
      fork
        monitor();
        flow_control();
      join_none
      repeat (4) @(negedge vif.clock);
      vif.reset = 0;
      vif.valid = 0;
      // Invalid target metadata is legal when no transaction is offered.
      for (int bitnum = 0; bitnum < 7; bitnum++) begin
        vif.target = 1 << bitnum;
        @(negedge vif.clock);
        vif.target = 0;
        @(negedge vif.clock);
      end
      vif.reset  = 1;
      vif.valid  = 1;
      vif.target = 0;
      repeat (2) @(negedge vif.clock);
      vif.reset = 0;
      vif.valid = 0;
      vif.target = 1;
      flow = 1;
      for (int packet = 0; packet < 32; packet++) begin
        vif.target = packet % 4 == 0 ? 64 : packet % 4;
        vif.valid  = 1;
        for (int bitnum = 0; bitnum < 389; bitnum++) vif.data[bitnum] = $urandom_range(0, 1);
        do begin
          @(posedge vif.clock);
          if (vif.ready) begin
            @(negedge vif.clock);
            break;
          end
          @(negedge vif.clock);
        end while (1);

      end
      vif.valid = 0;
      flow = 0;
      while (checked != 32) @(negedge vif.clock);
      repeat (8) @(negedge vif.clock);
      `uvm_info(
          "CHECKS",
          "32 real 389-bit CHI DAT flits serialized to 13 mesh chunks and reassembled under backpressure",
          UVM_LOW)
    endtask
  endclass
endpackage
