class env extends uvm_env;
  `uvm_component_utils(env)
  virtual preflight_control_if ctl;
  virtual stream_if #(`PF_CMD_WIDTH) cmd;
  virtual stream_if #(`PF_OUT_WIDTH) prepared;
  virtual stream_if #(`PF_AUTH_WIDTH) authorization;
  virtual stream_if #(`PF_PERMIT_WIDTH) permission;
  virtual stream_if #(`PF_PTE_WIDTH) pte_req;
  virtual stream_if #(`PF_PTERESP_WIDTH) pte_resp;
  result_board #(result_item, 1) scoreboard;
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
  covergroup behavior with function sample (int error, bit writing, bit last, bit pte, bit stall);
    option.per_instance = 1;
    outcomes: coverpoint error {bins all[] = {[0 : 6]};}
    direction: coverpoint writing;
    records: coverpoint last;
    authorization_kind: coverpoint pte;
    backpressure: coverpoint stall;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();

  endfunction
  function void verify_contract(bit ok, string message);
    verification_pkg::evidence::checks++;
    if (!ok) `uvm_fatal("PREFLIGHT", message)
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    model = preflight_ref_create(`PF_ADDRESS_BITS, `PF_BEAT_BYTES);
    scoreboard = result_board#(result_item, 1)::type_id::create("scoreboard", this);
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
    preflight_ref_program(model, root + ((va >> 30) & 511) * 8, ((root + 4096) >> 12) << 10 | 1, 0);
    preflight_ref_program(model, root + 4096 + ((va >> 21) & 511) * 8,
                          ((root + 8192) >> 12) << 10 | 1, 0);
    for (int i = 0; i < pages; i++) begin
      u64 a = (va & ~64'hfff) + i * 4096;
      // Deliberately noncontiguous 4KiB physical pages; no identity-map shortcut.
      preflight_ref_program(model, root + 8192 + ((a >> 12) & 511) * 8,
                            ((64'h800000 + i * 8192) >> 12) << 10 | flags, i == backend_error_page);
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
  function void final_phase(uvm_phase phase);
    preflight_ref_destroy(model);
    super.final_phase(phase);
  endfunction
  `include "agent/service.svh"
  `include "seq/scenario.svh"
endclass
