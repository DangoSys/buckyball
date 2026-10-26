class env extends uvm_env;
  `uvm_component_utils(env)
  virtual ddr_control_if ctl;
  virtual stream_if #(`DDR_REQ_WIDTH) req[2];
  virtual stream_if #(`DDR_RESP_WIDTH) resp[2];
  virtual stream_if #(`DDR_A_WIDTH) aw, ar;
  virtual stream_if #(`DDR_W_WIDTH) w;
  virtual stream_if #(`DDR_B_WIDTH) b;
  virtual stream_if #(`DDR_R_WIDTH) r;
  result_board #(result_item, 1) scoreboard;
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
  covergroup behavior with function sample (
      int client, bit writing, int mask_kind, bit error, int occupancy, bit reset_busy
  );
    option.per_instance = 1;
    clients: coverpoint client {bins all[] = {0, 1};}
    direction: coverpoint writing;
    masks: coverpoint mask_kind {bins all[] = {[0 : 2]};}
    errors: coverpoint error;
    occupancy_level: coverpoint occupancy {
      bins one = {1}; bins full = {`DDR_SLOTS}; ignore_bins other = {[2 : `DDR_SLOTS - 1]};
    }
    reset_cancellation: coverpoint reset_busy;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();

  endfunction
  function void verify_contract(bit ok, string message);
    verification_pkg::evidence::checks++;
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
    scoreboard = result_board#(result_item, 1)::type_id::create("scoreboard", this);
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
    verify_contract(a[`DF(A, LEN)] == `DDR_BEATS - 1 && a[`DF(A, SIZE)] == $clog2((`DDR_BEAT_BYTES)
                    ) && a[`DF(A, BURST)] == 1, "Wrong AXI burst geometry");
    verify_contract(a[`DF(A, LOCK)] == 0 && a[`DF(A, CACHE)] == 0 && a[`DF(A, PROT)] == 0 && a[
                    `DF(A, QOS)] == 0 && a[`DF(A, REGION)] == 0, "Unexpected AXI attributes");
    return key;
  endfunction
  function void final_phase(uvm_phase phase);
    ddr_ref_destroy(model);
    super.final_phase(phase);
  endfunction
  `include "agent/service.svh"
  `include "seq/scenario.svh"
endclass
