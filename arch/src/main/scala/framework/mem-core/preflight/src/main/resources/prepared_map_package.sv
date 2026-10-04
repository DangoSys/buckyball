`include "preflight_config.svh"
`define PF(K, F) `PF_``K``_``F``_OFFSET +: `PF_``K``_``F``_WIDTH
package prepared_map_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  typedef longint unsigned u64;
  import "DPI-C" function chandle pmap_ref_create(input int unsigned address_bits);
  import "DPI-C" function void pmap_ref_destroy(input chandle model);
  import "DPI-C" function void pmap_ref_reset(input chandle model);
  import "DPI-C" function void pmap_ref_reserve(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void pmap_ref_prepared(
    input chandle model,
    input int unsigned id,
    input u64 va,
    pa,
    input int unsigned bytes,
    write,
    last,
    error
  );
  import "DPI-C" function int unsigned pmap_ref_ready(
    input chandle model,
    input int unsigned id,
    fire,
    output int unsigned error,
    output u64 va
  );
  import "DPI-C" function void pmap_ref_query(
    input chandle model,
    input int unsigned valid,
    id,
    input u64 va,
    input int unsigned bytes,
    write,
    retiring,
    retiring_id,
    output int unsigned hit,
    output u64 pa,
    output int unsigned error
  );
  import "DPI-C" function void pmap_ref_release(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function int unsigned pmap_ref_pending(input chandle model);
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual prepared_map_control_if ctl;
    virtual stream_if #(`PF_RESERVE_WIDTH) reserve;
    virtual stream_if #(`PF_OUT_WIDTH) prepared;
    virtual stream_if #(`PF_MAP_READY_WIDTH) mapped;
    virtual stream_if #(`PF_RELEASE_WIDTH) retire;
    chandle model;
    bit notified[int];
    int status[int];
    int
        queries = 0,
        hits = 0,
        misses = 0,
        allocations = 0,
        notifications = 0,
        retired = 0,
        stalls = 0,
        dual = 0,
        cancelled = 0;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      timeout = 1ms;
    endfunction
    function void verify_contract(bit ok, string message);
      if (!ok) `uvm_fatal("PREPARED_MAP", message)
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      verify_contract(uvm_config_db#(virtual prepared_map_control_if)::get(this, "", "ctl", ctl),
                      "ctl missing");
      verify_contract(uvm_config_db#(virtual stream_if #(`PF_RESERVE_WIDTH))::get(
                      this, "", "reserve", reserve), "reserve missing");
      verify_contract(uvm_config_db#(virtual stream_if #(`PF_OUT_WIDTH))::get(
                      this, "", "prepared", prepared), "prepared missing");
      verify_contract(uvm_config_db#(virtual stream_if #(`PF_MAP_READY_WIDTH))::get(
                      this, "", "mapped", mapped), "mapped missing");
      verify_contract(uvm_config_db#(virtual stream_if #(`PF_RELEASE_WIDTH))::get(
                      this, "", "retire", retire), "retire missing");
      model = pmap_ref_create(`PF_ADDRESS_BITS);
    endfunction
    function void query_check(logic [`PF_QUERY_WIDTH-1:0] q, logic [`PF_RESULT_WIDTH-1:0] actual);
      int hit, error;
      u64 pa;
      pmap_ref_query(model, q[`PF(QUERY, VALID)], q[`PF(QUERY, ID)], q[`PF(QUERY, VA)], q[
                     `PF(QUERY, BYTES)], q[`PF(QUERY, WRITE)],
                     retire.sample.valid && retire.sample.ready, retire.sample.bits[`PF(RELEASE, ID)
                     ], hit, pa, error);
      verify_contract(actual[`PF(RESULT, HIT)] === 1'(hit) && actual[`PF(RESULT, PA)
                      ] === `PF_ADDRESS_BITS'(pa) && actual[`PF(RESULT, ERROR)] === 3'(error),
                      $sformatf(
                      "query id%0d VA%h bytes%0d expected hit%0d PA%h error%0d got %h",
                      q[
                      `PF(QUERY, ID)
                      ],
                      q[
                      `PF(QUERY, VA)
                      ],
                      q[
                      `PF(QUERY, BYTES)
                      ],
                      hit,
                      pa,
                      error,
                      actual
                      ));
      if (q[`PF(QUERY, VALID)]) begin
        queries++;
        if (hit) hits++;
        else misses++;
      end
    endfunction
    task monitor();
      forever begin
        @(ctl.sample);
        if (ctl.sample.reset) begin
          pmap_ref_reset(model);
          notified.delete();
          status.delete();
        end else begin
          query_check(ctl.sample.query_0, ctl.sample.result_0);
          query_check(ctl.sample.query_1, ctl.sample.result_1);
          if (ctl.sample.query_0[`PF(QUERY, VALID)] && ctl.sample.query_1[`PF(QUERY, VALID)])
            dual++;
          if (mapped.sample.valid) begin
            int error, id;
            u64 va;
            id = mapped.sample.bits[`PF(MAP_READY, ID)];
            verify_contract(pmap_ref_ready(model, id, mapped.sample.ready, error, va) == 1,
                            "unexpected notification");
            verify_contract(mapped.sample.bits[`PF(MAP_READY, ERROR)
                            ] === 3'(error) && mapped.sample.bits[`PF(MAP_READY, VA)] === va,
                            "notification mismatch");
            if (mapped.sample.ready) begin
              notified[id] = 1;
              status[id]   = error;
              notifications++;
            end else stalls++;
          end
          if (retire.sample.valid && retire.sample.ready) begin
            int id;
            id = retire.sample.bits[`PF(RELEASE, ID)];
            pmap_ref_release(model, id);
            notified.delete(id);
            retired++;
          end
          if (reserve.sample.valid && reserve.sample.ready) begin
            pmap_ref_reserve(model, reserve.sample.bits[`PF(RESERVE, ID)]);
            allocations++;
          end
          if (prepared.sample.valid && prepared.sample.ready)
            pmap_ref_prepared(model, prepared.sample.bits[`PF(OUT, ID)], prepared.sample.bits[
                              `PF(OUT, VA)], prepared.sample.bits[`PF(OUT, PA)],
                              prepared.sample.bits[`PF(OUT, BYTES)], prepared.sample.bits[
                              `PF(OUT, WRITE)], prepared.sample.bits[`PF(OUT, LAST)],
                              prepared.sample.bits[`PF(OUT, ERROR)]);
        end
      end
    endtask
    task reserve_tag(int id);
      @(negedge ctl.clock);
      reserve.bits = '0;
      reserve.bits[`PF(RESERVE, ID)] = id;
      reserve.valid = 1;
      do @(ctl.sample); while (!reserve.sample.ready);
      @(negedge ctl.clock);
      reserve.valid = 0;
    endtask
    task record(int id, u64 va, u64 pa, int bytes, bit write, bit last, int error = 0);
      @(negedge ctl.clock);
      prepared.bits = '0;
      prepared.bits[`PF(OUT, ID)] = id;
      prepared.bits[`PF(OUT, VA)] = va;
      prepared.bits[`PF(OUT, PA)] = pa;
      prepared.bits[`PF(OUT, BYTES)] = bytes;
      prepared.bits[`PF(OUT, WRITE)] = write;
      prepared.bits[`PF(OUT, LAST)] = last;
      prepared.bits[`PF(OUT, ERROR)] = error;
      prepared.valid = 1;
      do @(ctl.sample); while (!prepared.sample.ready);
      @(negedge ctl.clock);
      prepared.valid = 0;
    endtask
    function logic [`PF_QUERY_WIDTH-1:0] query(int id, u64 va, int bytes, bit write);
      logic [`PF_QUERY_WIDTH-1:0] q = '0;
      q[`PF(QUERY, VALID)] = 1;
      q[`PF(QUERY, ID)] = id;
      q[`PF(QUERY, VA)] = va;
      q[`PF(QUERY, BYTES)] = bytes;
      q[`PF(QUERY, WRITE)] = write;
      return q;
    endfunction
    task probes(logic [`PF_QUERY_WIDTH-1:0] a, logic [`PF_QUERY_WIDTH-1:0] b, int clocks = 3);
      @(negedge ctl.clock);
      ctl.query_0 = a;
      ctl.query_1 = b;
      repeat (clocks) @(ctl.sample);
      @(negedge ctl.clock);
      ctl.query_0 = '0;
      ctl.query_1 = '0;
    endtask
    task wait_ready(int id, int error = 0);
      while (!notified.exists(id)) @(negedge ctl.clock);
      verify_contract(status[id] == error, "wrong terminal status");
    endtask
    task retire_tag(int id, bit probe_release = 0);
      @(negedge ctl.clock);
      retire.bits = '0;
      retire.bits[`PF(RELEASE, ID)] = id;
      retire.valid = 1;
      if (probe_release) begin
        ctl.query_0 = query(id, 'h1010, 8, 0);
        ctl.query_1 = query(2, 'h4004, 8, 1);
      end
      do @(ctl.sample); while (!retire.sample.ready);
      @(negedge ctl.clock);
      retire.valid = 0;
      ctl.query_0  = '0;
      ctl.query_1  = '0;
    endtask
    task execute();
      ctl.reset = 1;
      ctl.hold_ready = 0;
      ctl.query_0 = '0;
      ctl.query_1 = '0;
      reserve.valid = 0;
      reserve.bits = '0;
      prepared.valid = 0;
      prepared.bits = '0;
      retire.valid = 0;
      retire.bits = '0;
      fork
        monitor();
      join_none
      repeat (3) @(negedge ctl.clock);
      ctl.reset = 0;
      reserve_tag(1);
      record(1, 'h1000, 'h8000, 4096, 0, 0);
      probes(query(1, 'h1010, 16, 0), query(1, 'h1010, 16, 1));
      @(negedge ctl.clock);
      ctl.hold_ready = 1;
      record(1, 'h2000, 'hc000, 4096, 0, 1);
      probes(query(1, 'h1010, 16, 0), query(1, 'h2010, 16, 0), 8);
      @(negedge ctl.clock);
      ctl.hold_ready = 0;
      wait_ready(1);
      probes(query(1, 'h1ffe, 2, 0), query(1, 'h2ffe, 2, 0));
      probes(query(1, 'h1fff, 2, 0), query(1, 'h3000, 4, 0));
      probes(query(1, 'h1000, 16, 1), query(1, 64'hffffffffffffffff, 2, 0));
      probes(query(1, 'h1000, 0, 0), query(99, 'h1000, 8, 0));
      reserve_tag(2);
      record(2, 'h4000, 'h9000, 64, 1, 0);
      record(2, 64'hffffffffffffffe0, (64'd1 << `PF_ADDRESS_BITS) - 32, 32, 1, 1);
      wait_ready(2);
      probes(query(1, 'h1f00, 16, 0), query(2, 'h4004, 16, 1));
      probes(query(2, 64'hfffffffffffffff8, 8, 1), query(2, 64'hfffffffffffffff8, 9, 1));
      retire_tag(1, 1);
      probes(query(1, 'h1010, 16, 0), query(2, 'h4004, 16, 1));
      reserve_tag(1);
      record(1, 'h6000, 'ha000, 128, 0, 1);
      wait_ready(1);
      probes(query(1, 'h1010, 16, 0), query(1, 'h6040, 64, 0));
      reserve_tag(3);
      record(3, 'h7000, 'hb000, 64, 0, 0);
      record(3, 'h8000, 0, 0, 0, 1, 3);
      wait_ready(3, 3);
      probes(query(3, 'h7004, 8, 0), query(3, 'h8000, 8, 0));
      reserve_tag(4);
      fork
        reserve_tag(5);
        begin
          repeat (6) begin
            @(negedge ctl.clock);
            verify_contract(!reserve.ready, "full table accepted fifth reserve");
          end
          record(4, 'h9000, 0, 0, 0, 1, 4);
          wait_ready(4, 4);
          retire_tag(4);
        end
      join
      for (int i = 0; i < 8; i++) begin
        record(5, 'h10000 + i * 4096, 'h30000 + (7 - i) * 4096, 32, 0, i == 7);
      end
      wait_ready(5);
      probes(query(5, 'h17018, 8, 0), query(5, 'h10008, 8, 0));
      retire_tag(1);
      retire_tag(2);
      retire_tag(3);
      retire_tag(5);
      reserve_tag(9);
      record(9, 'h9000, 'h19000, 64, 0, 1);
      // Reset cancels a terminal notification still under backpressure.
      @(negedge ctl.clock);
      ctl.hold_ready = 1;
      repeat (2) @(negedge ctl.clock);
      cancelled += pmap_ref_pending(model);
      ctl.reset = 1;
      repeat (3) @(negedge ctl.clock);
      ctl.reset = 0;
      ctl.hold_ready = 0;
      probes(query(9, 'h9000, 8, 0), query(1, 'h6000, 8, 0));
      repeat (3) @(negedge ctl.clock);
      verify_contract(pmap_ref_pending(model) == 0, "mapping remained after retirement/reset");
      verify_contract(hits > 0 && misses > 0 && dual > 0 && stalls >= 4 && cancelled == 1,
                      "required checks did not execute");
      `uvm_info(
          "PREPARED_MAP",
          $sformatf(
              "Checked %0d queries, %0d hits, %0d misses; reserve=%0d ready=%0d retire=%0d stalls=%0d dual=%0d reset_cancel=%0d",
              queries, hits, misses, allocations, notifications, retired, stalls, dual, cancelled),
          UVM_LOW)
    endtask
    function void final_phase(uvm_phase phase);
      pmap_ref_destroy(model);
      super.final_phase(phase);
    endfunction
  endclass
endpackage
