`include "profile.svh"
`define DF(K, F) `DDR_``K``_``F``_OFFSET +: `DDR_``K``_``F``_WIDTH
package ddr_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function chandle ddr_ref_create();
  import "DPI-C" function void ddr_ref_destroy(input chandle model);
  import "DPI-C" function void ddr_ref_program(
    input chandle model,
    input longint unsigned address,
    input bit [511:0] data
  );
  import "DPI-C" function int unsigned ddr_ref_read(
    input chandle model,
    input longint unsigned address,
    output bit [511:0] data
  );
  import "DPI-C" function int unsigned ddr_ref_write(
    input chandle model,
    input longint unsigned address,
    input bit [511:0] data,
    input longint unsigned mask
  );
  typedef longint unsigned u64;
  typedef bit [`DDR_REQ_WIDTH-1:0] req_t;
  class result_item extends uvm_sequence_item;
    bit [511:0] data;
    bit error;
    int tag, client;
    `uvm_object_utils_begin(result_item)

      `uvm_field_int(data, UVM_DEFAULT)
      `uvm_field_int(error, UVM_DEFAULT)

      `uvm_field_int(tag, UVM_DEFAULT)
      `uvm_field_int(client, UVM_DEFAULT)

    `uvm_object_utils_end
    function new(string name = "result_item");
      super.new(name);
    endfunction
  endclass
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual ddr_control_if ctl;
    virtual stream_if #(`DDR_REQ_WIDTH) req[2];
    virtual stream_if #(`DDR_RESP_WIDTH) resp[2];
    virtual stream_if #(`DDR_A_WIDTH) aw, ar;
    virtual stream_if #(`DDR_W_WIDTH) w;
    virtual stream_if #(`DDR_B_WIDTH) b;
    virtual stream_if #(`DDR_R_WIDTH) r;
    keyed_scoreboard #(result_item) scoreboard;
    chandle model;
    typedef struct {
      req_t packet;
      int   client;
      bit   done;
    } command;
    typedef struct {
      int key, beat, serial;
      bit [511:0] line;
    } reading;
    typedef struct {int key, due, serial;} writing;
    typedef struct packed {
      bit [511:0] data;
      bit [63:0]  mask;
    } wline;
    command commands[int];
    reading reading_ids[int];
    writing writing_ids[int];
    int aw_ids[$], aw_keys[$], aw_serials[$];
    wline w_lines[$];
    wline assembling;
    int assembling_beat = 0;
    int read_error[u64], write_error[u64];
    bit hold_aw = 0, hold_w = 0, hold_ar = 0, hold_b = 0, hold_r = 0, hold_resp[2] = '{0, 0};
    bit r_active = 0, b_active = 0;
    int r_id = 0, b_id = 0, last_r_id = -1, last_r_serial = -1, r_cursor = 0, r_beats = 0;
    logic [`DDR_R_WIDTH-1:0] r_packet;
    logic [`DDR_B_WIDTH-1:0] b_packet;
    int
        cycle = 0,
        accepted = 0,
        returned[2] = '{0, 0},
        cancelled = 0,
        peak = 0,
        aw_count = 0,
        w_bursts = 0,
        ar_count = 0;
    int
        request_stalls = 0,
        response_stalls = 0,
        aw_stalls = 0,
        w_stalls = 0,
        interleaves = 0,
        error_results = 0;
    int invalid_case = -1;
    int b_reorders = 0, r_reorders = 0, max_b_serial = -1, max_r_serial = -1;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      timeout = 1ms;
    endfunction
    function void verify_contract(bit ok, string message);
      if (!ok) `uvm_fatal("DDR_CONTRACT", message)
    endfunction
    function bit [511:0] pattern(int seed);
      bit [511:0] data;
      for (int i = 0; i < 8; i++)
      data[i*64+:64]=64'h0123456789abcdef^(64'h0101010101010101*(seed+i))^(64'h1<<((seed+i*7)%64));
      return data;
    endfunction
    function void initialize(u64 address, int seed);
      ddr_ref_program(model, address, pattern(seed));
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      model = ddr_ref_create();
      scoreboard = keyed_scoreboard#(result_item)::type_id::create("scoreboard", this);
      verify_contract(uvm_config_db#(virtual ddr_control_if)::get(this, "", "ctl", ctl),
                      "Missing control");
      for (int c = 0; c < 2; c++) begin
        verify_contract(uvm_config_db#(virtual stream_if #(`DDR_REQ_WIDTH))::get(
                        this, "", $sformatf("req%0d", c), req[c]), "Missing request");
        verify_contract(uvm_config_db#(virtual stream_if #(`DDR_RESP_WIDTH))::get(
                        this, "", $sformatf("resp%0d", c), resp[c]), "Missing response");
      end
      verify_contract(uvm_config_db#(virtual stream_if #(`DDR_A_WIDTH))::get(this, "", "aw", aw),
                      "Missing AW");
      verify_contract(uvm_config_db#(virtual stream_if #(`DDR_A_WIDTH))::get(this, "", "ar", ar),
                      "Missing AR");
      verify_contract(uvm_config_db#(virtual stream_if #(`DDR_W_WIDTH))::get(this, "", "w", w),
                      "Missing W");
      verify_contract(uvm_config_db#(virtual stream_if #(`DDR_B_WIDTH))::get(this, "", "b", b),
                      "Missing B");
      verify_contract(uvm_config_db#(virtual stream_if #(`DDR_R_WIDTH))::get(this, "", "r", r),
                      "Missing R");
      void'(uvm_config_db#(int)::get(this, "", "invalid_case", invalid_case));
    endfunction
    function int address_owner(u64 address, bit write);
      int found = -1;
      foreach (commands[key])
      if (commands[key].packet[
          `DF(REQ, ADDR)
          ] == address && commands[key].packet[
          `DF(REQ, WRITE)
          ] == write) begin
        verify_contract(found == -1, "Ambiguous fixture address");
        found = key;
      end
      verify_contract(found != -1, "AXI address without accepted line command");
      return found;
    endfunction
    function int observe_address(logic [`DDR_A_WIDTH-1:0] a, bit write);
      int key, id;
      verify_contract(!$isunknown(a), "Unknown AXI address payload");
      key = address_owner(a[`DF(A, ADDR)], write);
      id  = a[`DF(A, ID)];
      verify_contract(id >= `DDR_ID_BASE && id < `DDR_ID_BASE + `DDR_SLOTS,
                      "AXI ID outside configured namespace");
      verify_contract((id - `DDR_ID_BASE) / `DDR_SLOTS_PER_CLIENT == commands[key].client,
                      "AXI ID crossed client namespace");
      verify_contract(a[`DF(A, LEN)] == `DDR_BEATS - 1 && a[`DF(A, SIZE)] == $clog2(
                      (`DDR_BEAT_BYTES)) && a[`DF(A, BURST)] == 1, "Wrong AXI burst geometry");
      verify_contract(a[`DF(A, LOCK)] == 0 && a[`DF(A, CACHE)] == 0 && a[`DF(A, PROT)] == 0 && a[
                      `DF(A, QOS)] == 0 && a[`DF(A, REGION)] == 0, "Unexpected AXI attributes");
      return key;
    endfunction
    task service();
      forever begin
        @(ctl.sample);
        if (!ctl.sample.reset) begin
          cycle++;
          for (int c = 0; c < 2; c++) begin
            if (req[c].sample.valid && !req[c].sample.ready) request_stalls++;
            if (resp[c].sample.valid && !resp[c].sample.ready) response_stalls++;
            if (req[c].sample.valid && req[c].sample.ready) begin
              result_item expected = new;
              req_t q = req[c].sample.bits;
              int key = (c << 12) | int'(q[`DF(REQ, ID)]);
              bit [511:0] line;
              verify_contract(!commands.exists(key), "Accepted duplicate live client tag");
              verify_contract(ddr_ref_read(model, q[`DF(REQ, ADDR)], line),
                              "Uninitialized fixture line");
              commands[key] = '{q, c, 0};
              expected.set_transaction_id(key);
              expected.client = c;
              expected.tag = q[`DF(REQ, ID)];
              expected.error = q[`DF(REQ, WRITE)] ? write_error.exists(q[`DF(REQ, ADDR)]) :
                  read_error.exists(q[`DF(REQ, ADDR)]);
              expected.data = (q[`DF(REQ, WRITE)] || expected.error) ? '0 : line;
              scoreboard.expected_export.write(expected);
              accepted++;
              if (commands.num() > peak) peak = commands.num();
            end
          end
          if (aw.sample.valid && !aw.sample.ready) aw_stalls++;
          if (w.sample.valid && !w.sample.ready) w_stalls++;
          if (aw.sample.valid && aw.sample.ready) begin
            int key = observe_address(aw.sample.bits, 1);
            int id = aw.sample.bits[`DF(A, ID)];
            aw_ids.push_back(id);
            aw_keys.push_back(key);
            aw_serials.push_back(aw_count);
            aw_count++;
          end
          if (w.sample.valid && w.sample.ready) begin
            verify_contract(!$isunknown(w.sample.bits), "Unknown W payload");
            verify_contract(w.sample.bits[`DF(W, LAST)] == (assembling_beat == `DDR_BEATS - 1),
                            "Wrong WLAST");
            assembling.data[assembling_beat*`DDR_DATA_BITS+:`DDR_DATA_BITS] = w.sample.bits[
            `DF(W, DATA)
            ];
            assembling.mask[assembling_beat*(`DDR_BEAT_BYTES)+:(`DDR_BEAT_BYTES)] = w.sample.bits[
            `DF(W, STRB)
            ];
            if (assembling_beat == `DDR_BEATS - 1) begin
              w_lines.push_back(assembling);
              assembling = '0;
              assembling_beat = 0;
              w_bursts++;
            end else assembling_beat++;
          end
          if (aw_ids.size() && w_lines.size()) begin
            int   id = aw_ids.pop_front();
            int   key = aw_keys.pop_front();
            int   serial = aw_serials.pop_front();
            wline line = w_lines.pop_front();
            req_t q = commands[key].packet;
            verify_contract(!writing_ids.exists(id) && !reading_ids.exists(id),
                            "AXI ID reused while active");
            verify_contract(line.data === q[`DF(REQ, DATA)] && line.mask === q[`DF(REQ, MASK)],
                            "AW/W order, payload or byte mask mismatch");
            verify_contract(ddr_ref_write(model, q[`DF(REQ, ADDR)], line.data, line.mask),
                            "Write outside initialized DDR");
            writing_ids[id] = '{key, cycle + 7, serial};
          end
          if (ar.sample.valid && ar.sample.ready) begin
            int key = observe_address(ar.sample.bits, 0);
            int id = ar.sample.bits[`DF(A, ID)];
            bit [511:0] line;
            verify_contract(!reading_ids.exists(id) && !writing_ids.exists(id),
                            "AXI ID reused while active");
            verify_contract(ddr_ref_read(model, ar.sample.bits[`DF(A, ADDR)], line),
                            "Read outside initialized DDR");
            reading_ids[id] = '{key, 0, ar_count, line};
            ar_count++;
          end
          if (r.sample.valid && r.sample.ready) begin
            int key = reading_ids[r_id].key;
            r_beats++;
            if (last_r_id >= 0 && last_r_id != r_id && reading_ids.exists(
                    last_r_id
                ) && reading_ids[last_r_id].serial == last_r_serial)
              interleaves++;
            last_r_id = r_id;
            last_r_serial = reading_ids[r_id].serial;
            if (reading_ids[r_id].beat == `DDR_BEATS - 1) begin
              if (reading_ids[r_id].serial < max_r_serial) r_reorders++;
              else max_r_serial = reading_ids[r_id].serial;
              commands[key].done = 1;
              reading_ids.delete(r_id);
            end else reading_ids[r_id].beat++;
            r_active = 0;
            r_cursor = (r_id + 1) % 16;
          end
          if (b.sample.valid && b.sample.ready) begin
            if (writing_ids[b_id].serial < max_b_serial) b_reorders++;
            else max_b_serial = writing_ids[b_id].serial;
            commands[writing_ids[b_id].key].done = 1;
            writing_ids.delete(b_id);
            b_active = 0;
          end
          for (int c = 0; c < 2; c++)
          if (resp[c].sample.valid && resp[c].sample.ready) begin
            result_item actual = new;
            int key = (c << 12) | int'(resp[c].sample.bits[`DF(RESP, ID)]);
            verify_contract(!$isunknown(resp[c].sample.bits) && commands.exists(key
                            ) && commands[key].done,
                            "Early/unknown line response before final R or B");
            actual.set_transaction_id(key);
            actual.client = c;
            actual.tag = resp[c].sample.bits[`DF(RESP, ID)];
            actual.data = resp[c].sample.bits[`DF(RESP, DATA)];
            actual.error = resp[c].sample.bits[`DF(RESP, ERROR)];
            scoreboard.actual_export.write(actual);
            if (actual.error) error_results++;
            returned[c]++;
            commands.delete(key);
          end
        end
        @(negedge ctl.clock);
        aw.ready = !ctl.reset && !hold_aw && cycle % 5 != 0;
        w.ready  = !ctl.reset && !hold_w && cycle % 7 >= 2;
        ar.ready = !ctl.reset && !hold_ar && cycle % 4 != 0;
        for (int c = 0; c < 2; c++) resp[c].ready = !ctl.reset && !hold_resp[c] && cycle % 5 != c;
        if (!r_active && !hold_r)
          for (int off = 0; off < 16; off++) begin
            int id = (r_cursor + off) % 16;
            if (!r_active && reading_ids.exists(id)) begin
              int key = reading_ids[id].key;
              u64 address = commands[key].packet[`DF(REQ, ADDR)];
              int beat = reading_ids[id].beat;
              r_active = 1;
              r_id = id;
              r_packet = '0;
              r_packet[`DF(R, ID)] = id;
              r_packet[`DF(R, DATA)] = reading_ids[id].line[beat*`DDR_DATA_BITS+:`DDR_DATA_BITS];
              r_packet[`DF(R, LAST)] = beat == `DDR_BEATS - 1;
              if (read_error.exists(address) && beat == read_error[address])
                r_packet[`DF(R, RESP)] = beat % 2 ? 3 : 2;
            end
          end
        if (!b_active && !hold_b)
          for (int id = 15; id >= 0; id--)
          if (!b_active && writing_ids.exists(id) && writing_ids[id].due <= cycle) begin
            b_active = 1;
            b_id = id;
            b_packet = '0;
            b_packet[`DF(B, ID)] = id;
            if (write_error.exists(commands[writing_ids[id].key].packet[`DF(REQ, ADDR)]))
              b_packet[
              `DF(B, RESP)
              ] = write_error[commands[writing_ids[id].key].packet[
              `DF(REQ, ADDR)
              ]];
          end
        r.valid = !ctl.reset && r_active;
        r.bits  = r_packet;
        b.valid = !ctl.reset && b_active;
        b.bits  = b_packet;
      end
    endtask
    task submit(int client, int id, u64 address, bit write, bit [63:0] mask = '1);
      req_t q = '0;
      q[`DF(REQ, ID)] = id;
      q[`DF(REQ, ADDR)] = address;
      q[`DF(REQ, WRITE)] = write;
      q[`DF(REQ, DATA)] = pattern(id + client * 17 + int'(address >> 6));
      q[`DF(REQ, MASK)] = mask;
      @(negedge ctl.clock);
      req[client].bits  = q;
      req[client].valid = 1;
      do @(req[client].sample); while (!req[client].sample.ready);
      @(negedge ctl.clock);
      req[client].valid = 0;
    endtask
    task batch(u64 base, bit write);
      for (int c = 0; c < 2; c++)
        for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++)
          initialize(base + c * 4096 + i * 64, i + c * 7 + int'(base >> 6));
      fork
        begin
          for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++)
          submit(0, 12'h800 + i, base + i * 64, write,
                 i == 0 ? 0 : i == 1 ? '1 : i == 2 ? 64'h5555555555555555 : 64'haaaaaaaaaaaaaaaa);
        end
        begin
          for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++)
          submit(1, 12'h800 + i, base + 4096 + i * 64, write);
        end
      join
    endtask
    task drain();
      wait (commands.num() == 0);
      @(ctl.sample);
      verify_contract(
          !reading_ids.num()&&!writing_ids.num()&&!aw_ids.size()&&!w_lines.size()&&!assembling_beat&&!r_active&&!b_active,
          "Backend did not drain");
    endtask
    task joint_reset();
      @(negedge ctl.clock);
      ctl.reset = 1;
      req[0].valid = 0;
      req[1].valid = 0;
      repeat (3) @(ctl.sample);
      cancelled += commands.num();
      commands.delete();
      reading_ids.delete();
      writing_ids.delete();
      aw_ids.delete();
      aw_keys.delete();
      aw_serials.delete();
      w_lines.delete();
      scoreboard.expected.delete();
      scoreboard.actual.delete();
      assembling = '0;
      assembling_beat = 0;
      r_active = 0;
      b_active = 0;
      @(negedge ctl.clock);
      ctl.reset = 0;
    endtask
    function int finished_commands();
      int count = 0;
      foreach (commands[key]) if (commands[key].done) count++;
      return count;
    endfunction
    task invalid_scenario();
      int id, beats;
      bit aw_seen;
      if (invalid_case == 0) begin
        submit(0, 0, 'h10001, 0);
      end else if (invalid_case == 1) begin
        @(negedge ctl.clock);
        r.bits = '0;
        r.bits[`DF(R, ID)] = 0;
        r.valid = 1;
      end else if (invalid_case == 2 || invalid_case == 6) begin
        aw.ready = 1;
        w.ready  = invalid_case == 6;
        submit(0, 0, 'h10000, 1);
        beats   = 0;
        aw_seen = 0;
        do begin
          @(ctl.sample);
          if (aw.sample.valid && aw.sample.ready) begin
            id = aw.sample.bits[`DF(A, ID)];
            aw_seen = 1;
          end
          if (w.sample.valid && w.sample.ready) beats++;
        end while (!aw_seen || (invalid_case == 6 && beats != `DDR_BEATS));
        @(negedge ctl.clock);
        b.bits = '0;
        b.bits[`DF(B, ID)] = id;
        b.bits[`DF(B, RESP)] = invalid_case == 6 ? 1 : 0;
        b.valid = 1;
      end else begin
        ar.ready = 1;
        submit(0, 0, 'h10000, 0);
        do @(ctl.sample); while (!ar.sample.valid || !ar.sample.ready);
        id = ar.sample.bits[`DF(A, ID)];
        for (int beat = 0; beat < (invalid_case == 4 ? `DDR_BEATS : 1); beat++) begin
          @(negedge ctl.clock);
          r.bits = '0;
          r.bits[`DF(R, ID)] = id;
          r.bits[`DF(R, LAST)] = invalid_case == 3;
          r.bits[`DF(R, RESP)] = invalid_case == 5 ? 1 : 0;
          r.valid = 1;
          do @(r.sample); while (!r.sample.ready);
        end
      end
      repeat (8) @(ctl.sample);
      `uvm_fatal("MISSING_ASSERTION", "Invalid protocol did not trigger expected DUT assertion")
    endtask
    task execute();
      int before_aw, before_w, before_ar, before_r;
      int initial_returned;
      ctl.reset = 1;
      req[0].valid = 0;
      req[1].valid = 0;
      req[0].bits = '0;
      req[1].bits = '0;
      resp[0].ready = 0;
      resp[1].ready = 0;
      aw.ready = 0;
      ar.ready = 0;
      w.ready = 0;
      b.valid = 0;
      b.bits = '0;
      r.valid = 0;
      r.bits = '0;
      repeat (5) @(negedge ctl.clock);
      ctl.reset = 0;
      if (invalid_case >= 0) begin
        invalid_scenario();
        return;
      end
      fork
        service();
      join_none
      // W must progress even when AW has never been accepted; all eight slots stay live until B.
      hold_aw = 1;
      hold_b = 1;
      hold_resp[0] = 1;
      batch('h10000, 1);
      wait (w_bursts == `DDR_SLOTS);
      verify_contract(aw_count == 0 && returned[0] == 0 && returned[1] == 0,
                      "W-before-AW or B-delayed behavior failed");
      hold_aw = 0;
      wait (aw_count == `DDR_SLOTS);
      initialize('h18000, 99);
      fork
        submit(0, 12'h800, 'h18000, 1);
        begin
          hold_b = 0;
          wait (returned[1] == `DDR_SLOTS_PER_CLIENT);
          verify_contract(returned[0] == 0, "Client0 backpressure leaked");
          hold_resp[0] = 0;
        end
      join
      drain();
      for (int c = 0; c < 2; c++)
        for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++)
          submit(c, 12'h900 + i, 'h10000 + c * 4096 + i * 64, 0);
      drain();
      // Conversely allow addresses to run ahead of every W beat.
      before_aw = aw_count;
      before_w = w_bursts;
      hold_w = 1;
      batch('h20000, 1);
      wait (aw_count == before_aw + `DDR_SLOTS);
      verify_contract(w_bursts == before_w, "AW-before-W stalled ordering failed");
      hold_w = 0;
      drain();
      // Concurrent reads and writes: hold B until all independent reads return.
      hold_b = 1;
      initial_returned = returned[0] + returned[1];
      for (int c = 0; c < 2; c++)
        for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++)
          initialize('h30000 + c * 4096 + i * 64, 30 + i);
      fork
        begin
          for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++)
          submit(0, 12'ha00 + i, 'h30000 + i * 64, i[0]);
        end
        begin
          for (int i = 0; i < `DDR_SLOTS_PER_CLIENT; i++)
          submit(1, 12'ha00 + i, 'h31000 + i * 64, i[0]);
        end
      join
      wait (returned[0] + returned[1] == initial_returned + 2 * ((`DDR_SLOTS_PER_CLIENT + 1) / 2));
      hold_b = 0;
      drain();
      for (int i = 0; i < 4; i++) begin
        initialize('h40000 + i * 64, 40 + i);
        read_error['h40000+i*64] = i % `DDR_BEATS;
        submit(i % 2, 12'hb00 + i, 'h40000 + i * 64, 0);
      end
      initialize('h41000, 44);
      write_error['h41000] = 2;
      submit(1, 12'hb10, 'h41000, 1);
      initialize('h41040, 45);
      write_error['h41040] = 3;
      submit(0, 12'hb10, 'h41040, 1);
      drain();
      before_ar = ar_count;
      hold_r = 1;
      batch('h48000, 0);
      wait (ar_count == before_ar + `DDR_SLOTS);
      r_cursor = `DDR_ID_BASE;
      hold_r   = 0;
      drain();
      // Each byte strobe is independently checked, then read back through the reference.
      for (int i = 0; i < 64; i++) begin
        initialize('h50000 + i * 64, 50 + i);
        submit(i % 2, 12'hc00 + i, 'h50000 + i * 64, 1, 64'h1 << i);
        drain();
        submit(i % 2, 12'hd00 + i, 'h50000 + i * 64, 0);
        drain();
      end
      // Coordinated reset cancels queued AW-only, W-only and stalled line completions.
      before_aw = aw_count;
      hold_w = 1;
      batch('h60000, 1);
      wait (aw_count == before_aw + `DDR_SLOTS);
      joint_reset();
      hold_w   = 0;
      before_w = w_bursts;
      hold_aw  = 1;
      batch('h70000, 1);
      wait (w_bursts == before_w + `DDR_SLOTS);
      joint_reset();
      hold_aw = 0;
      before_ar = ar_count;
      before_r = r_beats;
      hold_r = 1;
      batch('h78000, 0);
      wait (ar_count == before_ar + `DDR_SLOTS);
      hold_r = 0;
      wait (r_beats >= before_r + 1);
      joint_reset();
      hold_resp[0] = 1;
      hold_resp[1] = 1;
      batch('h80000, 0);
      do @(ctl.sample); while (finished_commands() != `DDR_SLOTS);
      joint_reset();
      hold_resp[0] = 0;
      hold_resp[1] = 0;
      batch('h90000, 0);
      drain();
      verify_contract(
          peak==`DDR_SLOTS&&(`DDR_BEATS==1?interleaves==0:interleaves>0)&&b_reorders>0&&r_reorders>0&&request_stalls>0&&response_stalls>0&&aw_stalls>0&&w_stalls>0&&error_results==6&&cancelled==4*`DDR_SLOTS,
          "Missing concurrency, stalls, errors or reset coverage");
      verify_contract(accepted == returned[0] + returned[1] + cancelled, "Lost line response");
      `uvm_info(
          "DDR_PASS",
          $sformatf(
              "accepted=%0d returned=%0d/%0d cancelled=%0d peak=%0d errors=%0d readInterleave=%0d R/Breorder=%0d/%0d stalls req/resp/AW/W=%0d/%0d/%0d/%0d; full64B payload/mask/tag and all queues drained",
              accepted, returned[0], returned[1], cancelled, peak, error_results, interleaves,
              r_reorders, b_reorders, request_stalls, response_stalls, aw_stalls, w_stalls),
          UVM_LOW)
    endtask
    function void final_phase(uvm_phase phase);
      ddr_ref_destroy(model);
      super.final_phase(phase);
    endfunction
  endclass
endpackage
