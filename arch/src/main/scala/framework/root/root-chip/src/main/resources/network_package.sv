package mesh_network_pkg;
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
      timeout = 1ms;
      if (!uvm_config_db#(virtual mesh_fabric_if)::get(this, "", "vif", vif))
        `uvm_fatal("VIF", "fabric interface missing")
    endfunction
    typedef bit [33:0] golden_beat;
    typedef golden_beat beat_queue[$];
    task route_matrix();
      beat_queue expected[4][4][4];
      int submitted[4][4];
      int checked = 0;
      bit done;
      bit [31:0] payload;
      for (int i = 0; i < 4; i++) for (int v = 0; v < 4; v++) submitted[i][v] = 0;
      for (int tick = 0; tick < 60000; tick++) begin
        for (int source = 0; source < 4; source++)
        for (int v = 0; v < 4; v++) begin
          int index, destination, pattern;
          index = submitted[source][v];
          destination = index / (64 * 3);
          pattern = (index / 3) % 64;
          payload = 32'h1 << (pattern % 32);
          if (pattern >= 32) payload = ~payload;
          payload=payload^(source*32'h9e3779b9)^(v*32'h7f4a7c15)^(destination*32'h85ebca6b);
          vif.valid[source][v] = index < 768;
          vif.x[source][v] = destination % 2;
          vif.y[source][v] = destination / 2;
          vif.data[source][v] = payload;
          vif.head[source][v] = index % 3 == 0;
          vif.tail[source][v] = index % 3 == 2;
          vif.out_ready[source][v] = tick % 17 >= ((source + v) % 5);
        end
        @(posedge vif.clock);
        for (int source = 0; source < 4; source++)
        for (int v = 0; v < 4; v++)
        if (vif.valid[source][v] && vif.ready[source][v]) begin
          int destination;
          destination = vif.y[source][v] * 2 + vif.x[source][v];
          expected[source][destination][v].push_back(
              {vif.tail[source][v], vif.head[source][v], vif.data[source][v]});
          submitted[source][v]++;
        end
        for (int destination = 0; destination < 4; destination++)
        for (int v = 0; v < 4; v++)
        if (vif.out_valid[destination][v] && vif.out_ready[destination][v]) begin
          int source;
          golden_beat golden;
          source = vif.out_src_y[destination][v] * 2 + vif.out_src_x[destination][v];
          if (source >= 4 || !expected[source][destination][v].size())
            `uvm_fatal("MATRIX_EXTRA", "unmatched node/VC output")
          golden = expected[source][destination][v].pop_front();
          if({vif.out_tail[destination][v],vif.out_head[destination][v],vif.out_data[destination][v]}!==golden)
            `uvm_fatal("MATRIX_ORDER", "all-node source/destination/VC FIFO mismatch")
          checked++;
        end
        @(negedge vif.clock);
        done = checked == 12288;
        if (done) begin
          for (int source = 0; source < 4; source++)
          for (int destination = 0; destination < 4; destination++)
          for (int v = 0; v < 4; v++)
          if (expected[source][destination][v].size())
            `uvm_fatal("MATRIX_PENDING", "unmatched accepted all-node request");
          break;
        end
        if (tick == 59999)
          `uvm_fatal("MATRIX_TIMEOUT", "all-node walking/covering payload failed to drain")
      end
      for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++) begin
          vif.valid[i][v] = 0;
          vif.out_ready[i][v] = 1;
        end
      repeat (8) @(negedge vif.clock);
      `uvm_info(
          "CHECKS",
          "12288 all-node/all-VC walking+complement payload flits compared by per(source,destination,VC) FIFO including packet boundaries",
          UVM_LOW)
    endtask
    task isolated_paths();
      isolated_reset();
      // Reset cancels packet ownership; local input metadata is unconstrained while reset is asserted.
      vif.reset = 1;
      for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++) begin
          vif.valid[i][v] = 1;
          vif.head[i][v] = 0;
          vif.tail[i][v] = 1;
          vif.x[i][v] = i % 2;
          vif.y[i][v] = i / 2;
          vif.out_ready[i][v] = 1;
        end
      repeat (3) @(negedge vif.clock);
      for (int i = 0; i < 4; i++) for (int v = 0; v < 4; v++) vif.valid[i][v] = 0;
      vif.active = 0;
      repeat (3) @(negedge vif.clock);
      vif.reset  = 0;
      vif.active = 1;
      repeat (20) begin
        @(posedge vif.clock);
        for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++)
        if (vif.out_valid[i][v])
          `uvm_fatal("RESET_FINAL", "reset-time metadata survived as a packet");
        @(negedge vif.clock);
      end
      // Exercise each physical route independently: fill its buffers, pause packet
      // bodies while an owner is locked, then resume and compare every accepted flit.
      for (int source = 0; source < 4; source++)
        for (int path = 0; path < 6; path++)
          for (int vc = 0; vc < 4; vc++) begin
            int destination;
            int submitted = 0, checked = 0, next_cycle = 0;
            golden_beat expected[$];
            bit done = 0;
            destination = path < 4 ? path : path == 4 ? (source ^ 3) : (source ^ 1);
            for (int node = 0; node < 4; node++)
            for (int v = 0; v < 4; v++) begin
              vif.valid[node][v] = 0;
              vif.out_ready[node][v] = 1;
            end
            for (int tick = 0; tick < 300; tick++) begin
              vif.valid[source][vc] = submitted < 13 && tick >= next_cycle;
              vif.x[source][vc] = vif.valid[source][vc] ? destination % 2 : (destination ^ 3) % 2;
              vif.y[source][vc] = vif.valid[source][vc] ? destination / 2 : (destination ^ 3) / 2;
              vif.head[source][vc] = submitted % 4 == 0;
              vif.tail[source][vc] = submitted == 12 || submitted % 4 == 3;
              vif.data[source][vc]=32'h61900000|(source<<16)|(destination<<12)|(vc<<8)|submitted;
              vif.out_ready[destination][vc]=path>=4 || submitted>=12 || (tick>=45 && tick%13>=3);
              @(posedge vif.clock);
              if (vif.valid[source][vc] && vif.ready[source][vc]) begin
                expected.push_back({vif.tail[source][vc], vif.head[source][vc], vif.data[source][vc]
                                   });
                submitted++;
                next_cycle = tick + 1 + (submitted == 12 ? 30 : submitted % 4 == 1 ? 7 : 0);
              end
              for (int node = 0; node < 4; node++)
              for (int v = 0; v < 4; v++)
              if (vif.out_valid[node][v] && vif.out_ready[node][v]) begin
                golden_beat golden;
                if (node != destination || v != vc || !expected.size())
                  `uvm_fatal("PATH_EXTRA", "isolated path delivered unexpected flit")
                golden = expected.pop_front();
                if({vif.out_tail[node][v],vif.out_head[node][v],vif.out_data[node][v]}!==golden || vif.out_src_x[node][v]!=source%2 || vif.out_src_y[node][v]!=source/2)
                  `uvm_fatal("PATH_DATA", "isolated packet path mismatch")
                checked++;
              end
              @(negedge vif.clock);
              if (checked == 13) begin
                done = 1;
                break;
              end
            end
            if (!done || expected.size())
              `uvm_fatal("PATH_TIMEOUT", "isolated route failed to drain")
            vif.valid[source][vc] = 0;
            vif.out_ready[destination][vc] = 1;
            repeat (8) @(negedge vif.clock);
          end
      `uvm_info(
          "CHECKS",
          "1248 isolated-path packet flits checked with owner gaps and per-route FIFO saturation",
          UVM_LOW)
    endtask
    task isolated_reset();
      // Reset each physical horizontal link/VC after a packet head, before its tail.
      for (int source = 0; source < 4; source++)
        for (int vc = 0; vc < 4; vc++) begin
          repeat (12) @(negedge vif.clock);
          vif.valid[source][vc] = 1;
          vif.head[source][vc] = 1;
          vif.tail[source][vc] = 0;
          vif.x[source][vc] = (source ^ 1) % 2;
          vif.y[source][vc] = source / 2;
          vif.data[source][vc] = 32'hca110000 | (source << 8) | vc;
          @(posedge vif.clock);
          if (!vif.ready[source][vc])
            `uvm_fatal("RESET_CREDIT", "isolated reset launch did not have its initialized credit")
          @(negedge vif.clock);
          vif.reset = 1;
          vif.active = 0;
          vif.valid[source][vc] = 0;
          repeat (3) @(negedge vif.clock);
          vif.reset  = 0;
          vif.active = 1;
          repeat (12) begin
            @(posedge vif.clock);
            for (int i = 0; i < 4; i++)
            for (int v = 0; v < 4; v++)
            if (vif.out_valid[i][v])
              `uvm_fatal("RESET_ISOLATED", "cancelled per-VC flit survived reset");
            @(negedge vif.clock);
          end
        end
    endtask
    task reset_sweep();
      for (int step = 2; step < 13; step++) begin
        for (int source = 0; source < 4; source++)
        for (int v = 0; v < 4; v++) begin
          vif.valid[source][v] = 1;
          vif.x[source][v] = (source ^ 3) % 2;
          vif.y[source][v] = (source ^ 3) / 2;
          vif.head[source][v] = 1;
          vif.tail[source][v] = 1;
          vif.data[source][v] = 32'hcc000000 | (source << 8) | v;
          vif.out_ready[source][v] = 0;
        end
        repeat (step) @(negedge vif.clock);
        vif.reset  = 1;
        vif.active = 0;
        for (int source = 0; source < 4; source++)
        for (int v = 0; v < 4; v++) vif.valid[source][v] = 0;
        repeat (4) @(negedge vif.clock);
        vif.reset  = 0;
        vif.active = 1;
        for (int source = 0; source < 4; source++)
        for (int v = 0; v < 4; v++) vif.out_ready[source][v] = 1;
        repeat (20) begin
          @(posedge vif.clock);
          for (int source = 0; source < 4; source++)
          for (int v = 0; v < 4; v++)
          if (vif.out_valid[source][v])
            `uvm_fatal("RESET_STAGE", "cancelled pipeline-stage flit survived reset");
          @(negedge vif.clock);
        end
      end
    endtask
    task execute();
      int sent[4][4], received[4][4];
      bit complete;
      bit stalled[4][4];
      bit [31:0] held[4][4];
      vif.reset  = 1;
      vif.active = 0;
      for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++) begin
          sent[i][v] = 0;
          received[i][v] = 0;
          stalled[i][v] = 0;
          vif.valid[i][v] = 0;
          vif.x[i][v] = 1;
          vif.y[i][v] = 1;
          vif.out_ready[i][v] = 1;
        end
      repeat (4) @(negedge vif.clock);
      vif.reset = 0;
      for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++) begin
          vif.valid[i][v] = 1;
          vif.head[i][v]  = 1;
          vif.tail[i][v]  = 1;
          vif.data[i][v]  = 32'habcd;
        end
      repeat (4) begin
        @(posedge vif.clock);
        for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++)
        if (vif.ready[i][v] || vif.out_valid[i][v])
          `uvm_fatal("INACTIVE", "inactive local path accepted traffic");
        @(negedge vif.clock);
      end
      for (int i = 0; i < 4; i++) for (int v = 0; v < 4; v++) vif.valid[i][v] = 0;
      vif.active = 1;
      vif.out_ready[3][0] = 0;
      for (int tick = 0; tick < 1000; tick++) begin
        for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++) begin
          vif.valid[i][v] = sent[i][v] < 24;
          vif.head[i][v]  = sent[i][v] % 3 == 0;
          vif.tail[i][v]  = sent[i][v] % 3 == 2;
          vif.data[i][v]  = (i << 24) | (v << 16) | sent[i][v];
        end
        @(posedge vif.clock);
        for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++) if (vif.valid[i][v] && vif.ready[i][v]) sent[i][v]++;
        for (int destination = 0; destination < 4; destination++)
        for (int v = 0; v < 4; v++) begin
          if(stalled[destination][v]&&(!vif.out_valid[destination][v]||vif.out_data[destination][v]!==held[destination][v]))
            `uvm_fatal("STABLE", "network VC changed under stall")
          stalled[destination][v] = vif.out_valid[destination][v] && !vif.out_ready[destination][v];
          held[destination][v] = vif.out_data[destination][v];
        end
        for (int destination = 0; destination < 4; destination++)
        for (int v = 0; v < 4; v++)
        if (vif.out_valid[destination][v] && vif.out_ready[destination][v]) begin
          int source, seq;
          source = vif.out_data[destination][v] >> 24;
          seq = vif.out_data[destination][v] & 65535;
          if(destination!=3||source>=4||seq!=received[source][v]||((vif.out_data[destination][v]>>16)&255)!=v)
            `uvm_fatal("ORDER", "network lost, duplicated or reordered source/VC flit")
          received[source][v]++;
        end
        @(negedge vif.clock);
        if (tick == 350) begin
          for (int i = 0; i < 4; i++)
          for (int v = 1; v < 4; v++)
          if (received[i][v] != 24) `uvm_fatal("ISOLATION", "REQ blocked unrelated CHI protocol VC")
          vif.out_ready[3][0] = 1;
        end
        complete = 1;
        for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++) if (received[i][v] != 24) complete = 0;
        if (complete) break;
        if (tick == 999) `uvm_fatal("TIMEOUT", "multi-hop network did not drain")
      end
      for (int i = 0; i < 4; i++) for (int v = 0; v < 4; v++) vif.valid[i][v] = 0;
      repeat (10) @(negedge vif.clock);
      // Reset cancels buffered traffic and starts a fresh per-VC credit epoch.
      vif.out_ready[3][0] = 0;
      vif.valid[0][0] = 1;
      vif.head[0][0] = 1;
      vif.tail[0][0] = 1;
      vif.data[0][0] = 32'hdeadbeef;
      do @(posedge vif.clock); while (!vif.ready[0][0]);
      @(negedge vif.clock);
      vif.valid[0][0] = 0;
      repeat (12) @(negedge vif.clock);
      vif.reset  = 1;
      vif.active = 0;
      repeat (4) @(negedge vif.clock);
      vif.reset = 0;
      vif.active = 1;
      vif.out_ready[3][0] = 1;
      repeat (20) begin
        @(posedge vif.clock);
        for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++)
        if (vif.out_valid[i][v]) `uvm_fatal("RESET", "pre-reset flit survived coordinated reset");
        @(negedge vif.clock);
      end
      vif.valid[0][0] = 1;
      vif.data[0][0]  = 32'h5eedca11;
      do @(posedge vif.clock); while (!vif.ready[0][0]);
      @(negedge vif.clock);
      vif.valid[0][0] = 0;
      do @(posedge vif.clock); while (!vif.out_valid[3][0]);
      if (vif.out_data[3][0] !== 32'h5eedca11) `uvm_fatal("RESET", "fresh epoch payload mismatch");
      @(negedge vif.clock);
      repeat (8) @(negedge vif.clock);
      reset_sweep();
      route_matrix();
      isolated_paths();
      isolated_reset();
      // Reset cancels packet ownership; local input metadata is unconstrained while reset is asserted.
      vif.reset = 1;
      for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++) begin
          vif.valid[i][v] = 1;
          vif.head[i][v] = 0;
          vif.tail[i][v] = 1;
          vif.x[i][v] = i % 2;
          vif.y[i][v] = i / 2;
          vif.out_ready[i][v] = 1;
        end
      repeat (3) @(negedge vif.clock);
      for (int i = 0; i < 4; i++) for (int v = 0; v < 4; v++) vif.valid[i][v] = 0;
      vif.active = 0;
      repeat (3) @(negedge vif.clock);
      vif.reset  = 0;
      vif.active = 1;
      repeat (20) begin
        @(posedge vif.clock);
        for (int i = 0; i < 4; i++)
        for (int v = 0; v < 4; v++)
        if (vif.out_valid[i][v])
          `uvm_fatal("RESET_FINAL", "reset-time metadata survived as a packet");
        @(negedge vif.clock);
      end
      `uvm_info(
          "CHECKS",
          "384 multi-hop packet flits checked by source/VC; RSP/DAT/SNP finish while REQ blocked",
          UVM_LOW)
    endtask
  endclass
endpackage
