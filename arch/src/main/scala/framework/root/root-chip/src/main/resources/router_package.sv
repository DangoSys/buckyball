package mesh_router_pkg;
  import "DPI-C" function int mesh_route(
    int x,
    int y,
    int dst_x,
    int dst_y
  );
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual mesh_fabric_if vif;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      timeout = 300us;
      if (!uvm_config_db#(virtual mesh_fabric_if)::get(this, "", "vif", vif))
        `uvm_fatal("VIF", "fabric interface missing")
    endfunction
    task packet(int source, int direction, int channel, bit stall_head = 0, int length = 3);
      int dx, dy;
      dx = direction == 4 ? 0 : direction == 3 ? 2 : 1;
      dy = direction == 1 ? 0 : direction == 2 ? 2 : 1;
      vif.src_x[source][channel] = (source + direction) % 3;
      vif.src_y[source][channel] = (source + channel) % 3;
      for (int beat = 0; beat < length; beat++) begin
        if (beat == 1) begin
          vif.valid[source][channel] = 0;
          vif.out_ready[direction][channel] = 1;
          // VALID-low packet gap may carry unrelated metadata; ownership must survive.
          vif.x[source][channel] = dx == 0 ? 2 : 0;
          vif.y[source][channel] = dy == 0 ? 2 : 0;
          repeat (2) @(negedge vif.clock);
        end
        vif.valid[source][channel] = 1;
        vif.x[source][channel] = dx;
        vif.y[source][channel] = dy;
        vif.head[source][channel] = beat == 0;
        vif.tail[source][channel] = beat == length - 1;
        vif.data[source][channel] = $urandom;
        vif.out_ready[direction][channel] = beat == 0 && !stall_head;
        if (beat != 0 || stall_head)
          repeat (2) begin
            @(posedge vif.clock);
            if(vif.ready[source][channel]||vif.out_data[direction][channel]!==vif.data[source][channel])
              `uvm_fatal("STABLE", "packet beat unstable while blocked");
            @(negedge vif.clock);
          end
        vif.out_ready[direction][channel] = 1;
        @(posedge vif.clock);
        if (!vif.ready[source][channel] || !vif.out_valid[mesh_route(
                1, 1, dx, dy
            )][channel] || vif.out_data[direction][channel] !== vif.data[source][channel] ||
                vif.out_src_x[direction][channel] !== vif.src_x[source][channel] ||
                vif.out_src_y[direction][channel] !== vif.src_y[source][channel])
          `uvm_fatal("PACKET", "route or metadata mismatch")
        @(negedge vif.clock);
      end
      vif.valid[source][channel] = 0;
    endtask
    task execute();
      int counts  [5][4];
      int fired;
      bit accepted[5][4];
      vif.reset  = 1;
      vif.active = 0;
      for (int i = 0; i < 5; i++)
        for (int v = 0; v < 4; v++) begin
          vif.valid[i][v] = 0;
          vif.src_x[i][v] = (i + v) % 3;
          vif.src_y[i][v] = (i + v) / 3 % 3;
          vif.vc[i][v] = v;
          vif.x[i][v] = 1;
          vif.y[i][v] = 1;
          vif.head[i][v] = 1;
          vif.tail[i][v] = 1;
          vif.data[i][v] = 32'h100 + i;
          vif.out_ready[i][v] = 1;
        end
      repeat (4) @(negedge vif.clock);
      // Reset cancels transactions; metadata is unconstrained during reset.
      for (int i = 0; i < 5; i++)
        for (int v = 0; v < 4; v++) begin
          vif.valid[i][v] = 1;
          vif.x[i][v] = 3;
          vif.y[i][v] = 3;
          vif.vc[i][v] = (v + 1) % 4;
          vif.head[i][v] = 0;
        end
      for (int source = 0; source < 5; source++) begin
        for (int i = 0; i < 5; i++) for (int v = 0; v < 4; v++) vif.valid[i][v] = i == source;
        repeat (2) @(negedge vif.clock);
      end
      for (int i = 0; i < 5; i++) for (int v = 0; v < 4; v++) vif.valid[i][v] = 0;
      vif.reset = 0;
      repeat (4) @(negedge vif.clock);
      for (int i = 0; i < 5; i++)
        for (int v = 0; v < 4; v++) begin
          vif.x[i][v] = 3;
          vif.y[i][v] = 1;
        end
      repeat (2) @(negedge vif.clock);
      for (int i = 0; i < 5; i++)
        for (int v = 0; v < 4; v++) begin
          vif.x[i][v] = 1;
          vif.y[i][v] = 3;
        end
      repeat (2) @(negedge vif.clock);
      for (int i = 0; i < 5; i++)
        for (int v = 0; v < 4; v++) begin
          vif.x[i][v] = 1;
          vif.y[i][v] = 1;
          vif.vc[i][v] = v;
          vif.head[i][v] = 1;
        end
      for (int i = 0; i < 5; i++)
        for (int v = 0; v < 4; v++) begin
          counts[i][v] = 0;
          vif.valid[i][v] = 1;
        end
      for (int tick = 0; tick < 100; tick++) begin
        @(posedge vif.clock);
        fired = 0;
        for (int v = 0; v < 4; v++) begin
          fired = 0;
          for (int i = 0; i < 5; i++) begin
            accepted[i][v] = vif.ready[i][v];
            if (vif.ready[i][v]) begin
              fired++;
              counts[i][v]++;
              if (vif.out_data[0][v] !== vif.data[i][v])
                `uvm_fatal("DATA", "selected source mismatch");
            end
          end
          if (fired != 1)
            `uvm_fatal("CONTENTION", $sformatf("%0d inputs retired for one output", fired));
        end
        @(negedge vif.clock);
        for (int i = 0; i < 5; i++)
        for (int v = 0; v < 4; v++)
        if (accepted[i][v]) begin
          if (counts[i][v] == 20) vif.valid[i][v] = 0;
          else begin
            vif.data[i][v]  = $urandom;
            vif.src_x[i][v] = counts[i][v] % 3;
            vif.src_y[i][v] = (counts[i][v] / 3) % 3;
          end
        end
      end
      for (int i = 0; i < 5; i++)
        for (int v = 0; v < 4; v++) begin
          if (counts[i][v] != 20) `uvm_fatal("FAIR", "persistent contender starved");
          vif.valid[i][v] = 0;
        end
      // A blocked REQ must not stop independent RSP, DAT and SNP ports.
      vif.valid[4][0] = 1;
      vif.data[4][0] = 32'h104;
      vif.out_ready[0][0] = 0;
      for (int v = 1; v < 4; v++) vif.valid[0][v] = 1;
      repeat (20) begin
        @(posedge vif.clock);
        if (vif.ready[4][0]) `uvm_fatal("STALL", "blocked REQ retired")
        for (int v = 1; v < 4; v++)
        if (!vif.ready[0][v] || !vif.out_valid[0][v])
          `uvm_fatal("ISOLATION", "other CHI VC blocked by REQ")
        @(negedge vif.clock);
      end
      for (int v = 1; v < 4; v++) vif.valid[0][v] = 0;
      vif.out_ready[0][0] = 1;
      @(posedge vif.clock);
      if (!vif.ready[4][0] || vif.out_data[0][0] !== 32'h104)
        `uvm_fatal("RESUME", "stalled request was not preserved")
      @(negedge vif.clock);
      vif.valid[4][0] = 0;
      // Unlocked and empty output may be backpressured independently of a source.
      for (int direction = 0; direction < 5; direction++)
        for (int channel = 0; channel < 4; channel++) vif.out_ready[direction][channel] = 0;
      repeat (2) @(negedge vif.clock);
      for (int direction = 0; direction < 5; direction++)
        for (int channel = 0; channel < 4; channel++) vif.out_ready[direction][channel] = 1;
      // Each source seed and each offset from the round-robin cursor is exercised.
      for (int channel = 0; channel < 4; channel++)
        for (int direction = 0; direction < 5; direction++)
          for (int seed = 0; seed < 5; seed++)
            for (int offset = 0; offset < 5; offset++) begin
              packet(seed, direction, channel, offset % 2);
              packet((seed + 1 + offset) % 5, direction, channel);
            end
      for (int channel = 0; channel < 4; channel++)
        for (int direction = 0; direction < 5; direction++)
          for (int source = 0; source < 5; source++) packet(source, direction, channel, 0, 1);
      // Input metadata is unconstrained with VALID low; no packet may retire.
      for (int value = 0; value < 4; value++) begin
        for (int i = 0; i < 5; i++) for (int v = 0; v < 4; v++) vif.vc[i][v] = value;
        repeat (2) @(negedge vif.clock);
      end
      vif.reset = 1;
      repeat (4) @(negedge vif.clock);
      vif.reset = 0;
      for (int i = 0; i < 5; i++) for (int v = 0; v < 4; v++) vif.vc[i][v] = v;
      repeat (4) @(negedge vif.clock);
      `uvm_info(
          "CHECKS",
          "400 fair contended flits, 60 VC-isolation flits, 3100 routed packet flits; all cursor offsets and source coordinates with per-beat backpressure",
          UVM_LOW)
    endtask
  endclass
endpackage
