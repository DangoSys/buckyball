`ifdef PREFLIGHT_LINE_READ
`include "preflight64_config.svh"
`else
`include "preflight_config.svh"
`endif
`define PF(K, F) `PF_``K``_``F``_OFFSET +: `PF_``K``_``F``_WIDTH
package preflight_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  typedef longint unsigned u64;
  typedef bit [`PF_CMD_WIDTH-1:0] command_t;
  import "DPI-C" function chandle preflight_ref_create(
    input int unsigned address_bits,
    beat
  );
  import "DPI-C" function void preflight_ref_destroy(input chandle model);
  import "DPI-C" function void preflight_ref_program(
    input chandle model,
    input longint unsigned address,
    value,
    input int unsigned error
  );
  import "DPI-C" function void preflight_ref_policy(
    input chandle model,
    input int unsigned pte_valid,
    input longint unsigned pte,
    input int unsigned range_valid,
    input longint unsigned first,
    last
  );
  import "DPI-C" function int unsigned preflight_ref_submit(
    input chandle model,
    input int unsigned id,
    input longint unsigned base,
    input int unsigned rows,
    cols,
    span,
    col_stride,
    row_stride,
    write,
    mode,
    input longint unsigned root,
    input int unsigned privilege,
    sum,
    mxr
  );
  import "DPI-C" function int unsigned preflight_ref_count(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void preflight_ref_expected(
    input chandle model,
    input int unsigned id,
    index,
    output longint unsigned va,
    pa,
    output int unsigned bytes,
    write,
    last,
    error
  );
  import "DPI-C" function int unsigned preflight_ref_authorize(
    input chandle model,
    input int unsigned id,
    input longint unsigned pa,
    input int unsigned bytes,
    write,
    pte,
    privilege,
    output int unsigned allow
  );
  import "DPI-C" function int unsigned preflight_ref_pte(
    input chandle model,
    input int unsigned id,
    input longint unsigned address,
    output longint unsigned value,
    output int unsigned error
  );
  import "DPI-C" function int unsigned preflight_ref_pending(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void preflight_ref_retire(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void preflight_ref_reset(input chandle model);
  class result_item extends uvm_sequence_item;
    bit [`PF_OUT_WIDTH-1:0] packet;
    `uvm_object_utils_begin(result_item)
      `uvm_field_int(packet, UVM_DEFAULT)
    `uvm_object_utils_end
    function new(string name = "result_item");
      super.new(name);
    endfunction
  endclass
  class protocol_test extends ip_test;
    `uvm_component_utils(protocol_test)
    virtual preflight_control_if ctl;
    virtual stream_if #(`PF_CMD_WIDTH) cmd;
    virtual stream_if #(`PF_OUT_WIDTH) prepared;
    virtual stream_if #(`PF_AUTH_WIDTH) authorization;
    virtual stream_if #(`PF_PERMIT_WIDTH) permission;
    virtual stream_if #(`PF_PTE_WIDTH) pte_req;
    virtual stream_if #(`PF_PTERESP_WIDTH) pte_resp;
    keyed_scoreboard #(result_item) scoreboard;
    chandle model;
    bit live[int];
    int output_index[int];
    int cycle = 0, submitted = 0, retired = 0, segments = 0, failures[7] = '{default: 0}, peak = 0;
    int
        auth_count = 0,
        pte_count = 0,
        command_stalls = 0,
        output_stalls = 0,
        auth_stalls = 0,
        cancelled = 0;
    bit hold_output = 0, hold_permission = 0, auth_pending = 0, pte_pending = 0, pte_grant = 0;
    int auth_id, pte_id, grant_id, auth_due, pte_due;
    bit auth_allow, auth_pte;
    u64 grant_pa;
    logic [`PF_PERMIT_WIDTH-1:0] permit_packet;
    logic [`PF_PTERESP_WIDTH-1:0] pte_packet;
    u64 next_table = 'h10000;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      timeout = 1ms;
    endfunction
    function void verify_contract(bit ok, string message);
      if (!ok) `uvm_fatal("PREFLIGHT", message)
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      model = preflight_ref_create(`PF_ADDRESS_BITS, `PF_BEAT_BYTES);
      scoreboard = keyed_scoreboard#(result_item)::type_id::create("scoreboard", this);
      verify_contract(uvm_config_db#(virtual preflight_control_if)::get(this, "", "ctl", ctl),
                      "Missing control");
      verify_contract(uvm_config_db#(virtual stream_if #(`PF_CMD_WIDTH))::get(this, "", "cmd", cmd),
                      "Missing command");
      verify_contract(uvm_config_db#(virtual stream_if #(`PF_OUT_WIDTH))::get(
                      this, "", "prepared", prepared), "Missing output");
      verify_contract(uvm_config_db#(virtual stream_if #(`PF_AUTH_WIDTH))::get(
                      this, "", "authorization", authorization), "Missing authorization");
      verify_contract(uvm_config_db#(virtual stream_if #(`PF_PERMIT_WIDTH))::get(
                      this, "", "permission", permission), "Missing permission");
      verify_contract(uvm_config_db#(virtual stream_if #(`PF_PTE_WIDTH))::get(
                      this, "", "pte_req", pte_req), "Missing PTE request");
      verify_contract(uvm_config_db#(virtual stream_if #(`PF_PTERESP_WIDTH))::get(
                      this, "", "pte_resp", pte_resp), "Missing PTE response");
    endfunction
    function command_t request(int id, u64 base, int rows, int span, int row_stride, int cols = 1,
                               int col_stride = 0, bit write = 0, int mode = 0, u64 root = 0,
                               int privilege = 1, bit sum = 0, bit mxr = 0);
      command_t q = '0;
      q[`PF(CMD, ID)] = id;
      q[`PF(CMD, BASEVA)] = base;
      q[`PF(CMD, ROWS)] = rows;
      q[`PF(CMD, COLUMNS)] = cols;
      q[`PF(CMD, SPANBYTES)] = span;
      q[`PF(CMD, ROWSTRIDE)] = row_stride;
      q[`PF(CMD, COLUMNSTRIDE)] = col_stride;
      q[`PF(CMD, WRITE)] = write;
      q[`PF(CMD, MODE)] = mode;
      q[`PF(CMD, ROOTPPN)] = root;
      q[`PF(CMD, PRIVILEGE)] = privilege;
      q[`PF(CMD, SUM)] = sum;
      q[`PF(CMD, MXR)] = mxr;
      return q;
    endfunction
    function u64 mapping(u64 va, int pages, u64 flags = 'hc7, int backend_error_page = -1);
      u64 root = next_table;
      next_table += 'h3000;
      preflight_ref_program(model, root + ((va >> 30) & 511) * 8, ((root + 4096) >> 12) << 10 | 1,
                            0);
      preflight_ref_program(model, root + 4096 + ((va >> 21) & 511) * 8,
                            ((root + 8192) >> 12) << 10 | 1, 0);
      for (int i = 0; i < pages; i++) begin
        u64 a = (va & ~64'hfff) + i * 4096;
        // Deliberately noncontiguous 4KiB physical pages; no identity-map shortcut.
        preflight_ref_program(model, root + 8192 + ((a >> 12) & 511) * 8,
                              ((64'h800000 + i * 8192) >> 12) << 10 | flags,
                              i == backend_error_page);
      end
      return root >> 12;
    endfunction
    function void predict(command_t q);
      int id = q[`PF(CMD, ID)];
      int count;
      verify_contract(!live.exists(id), "Duplicate admitted command ID");
      verify_contract(preflight_ref_submit(
                      model,
                      id,
                      q[
                      `PF(CMD, BASEVA)
                      ],
                      q[
                      `PF(CMD, ROWS)
                      ],
                      q[
                      `PF(CMD, COLUMNS)
                      ],
                      q[
                      `PF(CMD, SPANBYTES)
                      ],
                      q[
                      `PF(CMD, COLUMNSTRIDE)
                      ],
                      q[
                      `PF(CMD, ROWSTRIDE)
                      ],
                      q[
                      `PF(CMD, WRITE)
                      ],
                      q[
                      `PF(CMD, MODE)
                      ],
                      q[
                      `PF(CMD, ROOTPPN)
                      ],
                      q[
                      `PF(CMD, PRIVILEGE)
                      ],
                      q[
                      `PF(CMD, SUM)
                      ],
                      q[
                      `PF(CMD, MXR)
                      ]
                      ) == 0, "Uninitialized PTE in reference fixture");
      live[id] = 1;
      output_index[id] = 0;
      submitted++;
      if (live.num() > peak) peak = live.num();
      count = preflight_ref_count(model, id);
      for (int i = 0; i < count; i++) begin
        result_item expected = new;
        u64 va, pa;
        int unsigned bytes, write, last, error;
        preflight_ref_expected(model, id, i, va, pa, bytes, write, last, error);
        expected.packet = '0;
        expected.packet[`PF(OUT, ID)] = id;
        expected.packet[`PF(OUT, VA)] = va;
        expected.packet[`PF(OUT, PA)] = pa;
        expected.packet[`PF(OUT, BYTES)] = bytes;
        expected.packet[`PF(OUT, WRITE)] = write;
        expected.packet[`PF(OUT, LAST)] = last;
        expected.packet[`PF(OUT, ERROR)] = error;
        expected.set_transaction_id((id << 4) | i);
        scoreboard.expected_export.write(expected);
      end
    endfunction
    task service();
      forever begin
        @(ctl.sample);
        if (!ctl.sample.reset) begin
          cycle++;
          if (cmd.sample.valid && !cmd.sample.ready) command_stalls++;
          if (prepared.sample.valid && !prepared.sample.ready) output_stalls++;
          if (authorization.sample.valid && !authorization.sample.ready) auth_stalls++;
          if (cmd.sample.valid && cmd.sample.ready) predict(cmd.sample.bits);
          if (permission.sample.valid && permission.sample.ready) begin
            verify_contract(auth_pending, "Unsolicited permission consumed");
            if (auth_pte && auth_allow) begin
              pte_grant = 1;
              grant_id  = auth_id;
            end
            auth_pending = 0;
          end
          if (authorization.sample.valid && authorization.sample.ready) begin
            int unsigned allow;
            int id = authorization.sample.bits[`PF(AUTH, ID)];
            verify_contract(!$isunknown(authorization.sample.bits) && !auth_pending,
                            "Overlapping/unknown authorization");
            verify_contract(preflight_ref_authorize(
                            model,
                            id,
                            authorization.sample.bits[
                            `PF(AUTH, PA)
                            ],
                            authorization.sample.bits[
                            `PF(AUTH, BYTES)
                            ],
                            authorization.sample.bits[
                            `PF(AUTH, WRITE)
                            ],
                            authorization.sample.bits[
                            `PF(AUTH, ISPTE)
                            ],
                            authorization.sample.bits[
                            `PF(AUTH, PRIVILEGE)
                            ],
                            allow
                            ) == 1, "Authorization lost snapshot, range or PTE S privilege");
            auth_pending = 1;
            auth_id = id;
            auth_allow = allow;
            auth_pte = authorization.sample.bits[`PF(AUTH, ISPTE)];
            grant_pa = authorization.sample.bits[`PF(AUTH, PA)];
            auth_due = cycle + 3;
            permit_packet = '0;
            permit_packet[`PF(PERMIT, ID)] = id;
            permit_packet[`PF(PERMIT, ALLOW)] = allow;
            auth_count++;
          end
          if (pte_resp.sample.valid && pte_resp.sample.ready) pte_pending = 0;
          if (pte_req.sample.valid && pte_req.sample.ready) begin
            u64 value;
            int unsigned error;
            verify_contract(pte_grant && !pte_pending && !$isunknown(pte_req.sample.bits),
                            "PTE accessed before authorization");
            verify_contract(pte_req.sample.bits[`PF(PTE, ADDR)] == grant_pa && !pte_req.sample.bits[
                            `PF(PTE, WRITE)] && pte_req.sample.bits[`PF(PTE, ATOMIC)
                            ] == 0 && !pte_req.sample.bits[`PF(PTE, ATOMICWORD)
                            ] && pte_req.sample.bits[`PF(PTE, MASK)] == 0,
                            "PTE request changed address or attempted writeback");
            verify_contract(preflight_ref_pte(model, grant_id, grant_pa, value, error) == 1,
                            "PTE address chain differs from architectural walker");
            pte_packet = '0;
            pte_packet[`PF(PTERESP, DATA)] = value;
            pte_packet[`PF(PTERESP, ERROR)] = error;
            pte_id = grant_id;
            pte_pending = 1;
            pte_due = cycle + 2 + (pte_count % 4);
            pte_grant = 0;
            pte_count++;
          end
          if (prepared.sample.valid) begin
            int id = prepared.sample.bits[`PF(OUT, ID)];
            verify_contract(live.exists(id) && !$isunknown(prepared.sample.bits),
                            "Unsolicited/unknown prepared plan");
            verify_contract(
                preflight_ref_pending(model, id
                ) == 0 && !(auth_pending && auth_id == id) && !(pte_pending && pte_id == id),
                "Partial plan exposed before all translations/permissions completed");
            if (prepared.sample.ready) begin
              result_item actual = new;
              actual.packet = prepared.sample.bits;
              actual.set_transaction_id((id << 4) | output_index[id]);
              scoreboard.actual_export.write(actual);
              output_index[id]++;
              segments++;
              if (prepared.sample.bits[`PF(OUT, LAST)]) begin
                failures[int'(prepared.sample.bits[`PF(OUT, ERROR)])]++;
                retired++;
                live.delete(id);
                output_index.delete(id);
                preflight_ref_retire(model, id);
              end
            end
          end
        end
        @(negedge ctl.clock);
        authorization.ready = !ctl.reset && !auth_pending && cycle % 5 >= 2;
        permission.valid = !ctl.reset && auth_pending && !hold_permission && cycle >= auth_due;
        permission.bits = permit_packet;
        pte_req.ready = !ctl.reset && !pte_pending && cycle % 4 != 0;
        pte_resp.valid = !ctl.reset && pte_pending && cycle >= pte_due;
        pte_resp.bits = pte_packet;
        prepared.ready = !ctl.reset && !hold_output && cycle % 7 >= 3;
      end
    endtask
    task submit(command_t q);
      @(negedge ctl.clock);
      cmd.bits  = q;
      cmd.valid = 1;
      do @(cmd.sample); while (!cmd.sample.ready);
      @(negedge ctl.clock);
      cmd.valid = 0;
      cmd.bits  = ~q;
    endtask
    task drain();
      wait (live.num() == 0);
      @(ctl.sample);
      verify_contract(!auth_pending && !pte_pending && !pte_grant,
                      "Preparation endpoints did not drain");
    endtask
    task execute();
      u64 root, root2;
      int prior;
      command_t q;
      ctl.reset = 1;
      cmd.valid = 0;
      cmd.bits = '0;
      prepared.ready = 0;
      authorization.ready = 0;
      permission.valid = 0;
      permission.bits = '0;
      pte_req.ready = 0;
      pte_resp.valid = 0;
      pte_resp.bits = '0;
      repeat (5) @(negedge ctl.clock);
      ctl.reset = 0;
      fork
        service();
      join_none
      // Current BERT maximum: 1024 dense 16B rows normalize to five pages.
      root = mapping('h400000f0, 9);
      root2 = mapping('h50000000, 9);
      hold_output = 1;
      submit(request(1, 'h400000f0, 1024, 16, 16, 1, 0, 0, 8, root));
      submit(request(2, 'h50000000, 2048, 16, 16, 1, 0, 1, 8, root2));  // exactly eight pages
      submit(request(3, 'h60000ff8, 1, 16, 16));  // alignment-expanded beat crosses page
      submit(request(4, 'h70000000, 1, 16, 16));
      wait (prepared.sample.valid);
      repeat (8) @(ctl.sample);
      fork
        submit(request(5, 'h70002000, 1, 16, 16));
        begin
          repeat (12) @(ctl.sample);
          hold_output = 0;
        end
      join
      drain();
      submit(request(6, 'h50000000, 2304, 16, 16, 1, 0, 0, 8, root2));
      drain();  // ninth page: one error, zero successful output
      submit(request(7, 'h60000000, 2, 16, 8192, 2, 4096));
      drain();  // sparse 2D, four extents
      submit(request(8, 'h60000000, 1, 16, 16));
      drain();  // progress immediately after capacity error
      root = mapping('h40010000, 1, 'h43);
      submit(request(9, 'h40010000, 1, 16, 16, 1, 0, 1, 8, root));
      drain();
      root = mapping('h40020000, 1, 'h7);
      submit(request(10, 'h40020000, 1, 16, 16, 1, 0, 0, 8, root));
      drain();
      root = mapping('h40030000, 1, 'hc7, 0);
      submit(request(11, 'h40030000, 1, 16, 16, 1, 0, 0, 8, root));
      drain();
      root = mapping('h40040000, 1);
      preflight_ref_policy(model, 1, (root << 12) + ((64'h40040000 >> 30) & 511) * 8, 0, 0, 0);
      submit(request(12, 'h40040000, 1, 16, 16, 1, 0, 0, 8, root));
      drain();
      preflight_ref_policy(model, 0, 0, 1, 'h800000, 'h80000f);
      submit(request(13, 'h40040000, 1, 16, 16, 1, 0, 0, 8, root));
      drain();
      preflight_ref_policy(model, 0, 0, 1, 'h71000000, 'h71000003);
      // Denied bytes precede the user span: the aligned transport range is still authorized.
      submit(request(14, 'h71000004, 1, 4, 4));
      drain();
      preflight_ref_policy(model, 0, 0, 0, 0, 0);
      submit(request(15, (64'h1 << `PF_ADDRESS_BITS) - 16, 1, 32, 32));
      drain();
      submit(request(16, 64'hfffffffffffffff8, 1, 16, 16));
      drain();
      submit(request(17, 'h80000000, 0, 16, 16));
      drain();
      q = request(18, 'h80000000, 1, 16, 16);
      q[`PF(CMD, MODE)] = 9;
      submit(q);
      drain();
      root = mapping('h40050000, 1, 'h59);
      submit(request(19, 'h40050000, 1, 16, 16, 1, 0, 0, 8, root, 1, 1, 1));
      drain();
      // External snapshot service captures policy at command acceptance, not from later live settings.
      hold_permission = 1;
      submit(request(20, 'h72000000, 1, 16, 16));
      preflight_ref_policy(model, 0, 0, 1, 'h72000000, 'h7200000f);
      wait (auth_pending);
      repeat (8) @(ctl.sample);
      hold_permission = 0;
      drain();
      preflight_ref_policy(model, 0, 0, 0, 0, 0);
      submit(request(21, 'h73000000, 1, 16, 256, 16, 16));
      drain();  // columns retains all 32 input bits
      q = request(22, 64'hfffffffffffff000, -1, -1, -1, -1, -1);
      submit(q);
      drain();
      // Write authorization covers actual bytes; read authorization covers transport padding.
      prior = failures[4];
      preflight_ref_policy(model, 0, 0, 1, 'h74000000, 'h74000003);
      submit(request(23, 'h74000004, 1, 4, 4, 1, 0, 1));
      drain();
      verify_contract(failures[4] == prior, "write was denied for untouched prefix padding");
      submit(request(24, 'h74000004, 1, 4, 4));
      drain();
      verify_contract(failures[4] == prior + 1, "read failed to authorize actual prefix overread");
      prior = failures[4];
      preflight_ref_policy(model, 0, 0, 1, 'h74000108, 'h7400013f);
      submit(request(25, 'h74000104, 1, 4, 4, 1, 0, 1));
      drain();
      verify_contract(failures[4] == prior, "write was denied for untouched tail padding");
      submit(request(26, 'h74000104, 1, 4, 4));
      drain();
      verify_contract(failures[4] == prior + 1, "read failed to authorize actual tail overread");
      prior = failures[4];
      preflight_ref_policy(model, 0, 0, 1, 'h74002ff0, 'h74002ffb);
      submit(request(27, 'h74002ffc, 1, 8, 8, 1, 0, 1));
      drain();
      verify_contract(failures[4] == prior, "cross-page write included untouched bytes");
      submit(request(28, 'h74002ffc, 1, 8, 8));
      drain();
      verify_contract(failures[4] == prior + 1, "cross-page read omitted padded transport bytes");
      preflight_ref_policy(model, 0, 0, 0, 0, 0);
      verify_contract(
          peak==4&&submitted==retired&&command_stalls>0&&output_stalls>0&&auth_stalls>0,
          "Missing concurrency/backpressure/drain coverage");
      for (int error = 1; error < 7; error++)
        verify_contract(failures[error] > 0, "Missing typed error scenario");
      verify_contract(scoreboard.checked == segments, "Not every output descriptor was compared");
      `uvm_info(
          "PREFLIGHT_PASS",
          $sformatf(
              "commands=%0d descriptors=%0d peak=%0d PTE=%0d auth=%0d errors shape/overflow/PF/AF/capacity/context=%0d/%0d/%0d/%0d/%0d/%0d stalls cmd/out/auth=%0d/%0d/%0d; all plans/permissions drained",
              retired, segments, peak, pte_count, auth_count, failures[1], failures[2], failures[3],
              failures[4], failures[5], failures[6], command_stalls, output_stalls, auth_stalls),
          UVM_LOW)
    endtask
    function void final_phase(uvm_phase phase);
      preflight_ref_destroy(model);
      super.final_phase(phase);
    endfunction
  endclass
endpackage
