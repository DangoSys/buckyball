package layernorm_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function int unsigned layernorm_case_count();
  import "DPI-C" function void layernorm_case_load(input int unsigned index);
  import "DPI-C" function int unsigned layernorm_case_field(input int unsigned field);
  import "DPI-C" function longint unsigned layernorm_case_word(
    input int unsigned role,
    row,
    high
  );
  import "DPI-C" function longint unsigned layernorm_bemu_cycles(
    input longint unsigned iter,
    input longint unsigned rs2,
    input int unsigned words[4096],
    input int unsigned eps
  );
  `include "layernorm_cycles.svh"
  function automatic bit [127:0] fixture(int role, row);
    return {layernorm_case_word(role, row, 1), layernorm_case_word(role, row, 0)};
  endfunction

  class layernorm_seq extends uvm_sequence #(bb_blink_cmd_item);
    `uvm_object_utils(layernorm_seq)
    int unsigned rob;
    function new(string name = "layernorm_seq");
      super.new(name);
    endfunction
    task body();
      bb_blink_cmd_item item = bb_blink_cmd_item::type_id::create("item");
      start_item(item);
      item.rob_id = rob;
      item.funct = `LAYERNORM_FUNCT7;
      item.iter = layernorm_case_field(0);
      item.rs2 = 64'(layernorm_case_field(1)) | (64'(layernorm_case_field(2)) << 16) |
          (64'(layernorm_case_field(3)) << 30) | (64'(layernorm_case_field(4)) << 44) |
          (64'(layernorm_case_field(5)) << 58);
      item.op1_bank = layernorm_case_field(6);
      item.op2_bank = layernorm_case_field(7);
      item.wr_bank = layernorm_case_field(8);
      item.op1_group = layernorm_case_field(9);
      item.op2_group = layernorm_case_field(10);
      item.wr_group = layernorm_case_field(11);
      item.op1_col = 1;
      item.op2_col = 1;
      item.wr_col = 1;
      item.wr_read_bank = item.wr_bank;
      item.wr_arch_bank = item.wr_bank;
      finish_item(item);
    endtask
  endclass

  class layernorm_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(layernorm_scoreboard)
    uvm_analysis_imp_stim #(bb_blink_cmd_item, layernorm_scoreboard) stim_imp;
    uvm_analysis_imp_cmd #(bb_blink_cmd_item, layernorm_scoreboard) cmd_imp;
    uvm_analysis_imp_read #(bb_blink_read_item, layernorm_scoreboard) read_imp;
    uvm_analysis_imp_write #(bb_blink_write_item, layernorm_scoreboard) write_imp;
    uvm_analysis_imp_resp #(bb_blink_resp_item, layernorm_scoreboard) resp_imp;
    bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;
    bb_blink_cmd_item command;
    int unsigned rows, w, lines, in_base, out_base, param_base, rms, value_class, backpressure;
    int unsigned bank[3], group[3];
    int writes, responses;
    // The bank rows each read port must request, in order.
    int unsigned addresses[2][$];
    time accepted;
    covergroup contract_cov;
      option.per_instance = 1;
      cp_w: coverpoint w {
        bins minimum = {16};
        bins narrow = {[17 : 31]};
        bins middle = {[32 : 128]};
        bins wide = {[129 : 512]};
        bins widest = {[513 : 2043]};
        bins layernorm_max = {2044};
        bins rms_max = {2048};
      }
      cp_tail: coverpoint (w % 4) {bins aligned = {0}; bins tail[] = {1, 2, 3};}
      cp_rms: coverpoint rms {bins layernorm = {0}; bins rmsnorm = {1};}
      cp_rows: coverpoint rows {
        bins one = {1}; bins few = {[2 : 8]}; bins many = {[9 : 64]}; bins whole_bank = {256};
      }
      cp_class: coverpoint value_class {
        bins random = {0};
        bins constant = {1};
        bins subnormal = {2};
        bins offset = {3};
        bins zeros = {4};
        bins wide = {5};
        bins zero_eps = {6};
        bins large_eps = {7};
      }
      cp_in_end: coverpoint (in_base + rows * lines == 1024) {
        bins inner = {0}; bins last_row = {1};
      }
      cp_out_end: coverpoint (out_base + rows * lines == 1024) {
        bins inner = {0}; bins last_row = {1};
      }
      cp_param_end: coverpoint (param_base + (rms ? lines : 2 * lines) + 1 == 1024) {
        bins inner = {0}; bins last_row = {1};
      }
      cp_in_base: coverpoint (in_base != 0) {bins zero = {0}; bins offset = {1};}
      cp_out_base: coverpoint (out_base != 0) {bins zero = {0}; bins offset = {1};}
      cp_param_base: coverpoint (param_base != 0) {bins zero = {0}; bins offset = {1};}
      cp_backpressure: coverpoint backpressure {bins ready = {0}; bins held = {1};}
      cx_rms_tail: cross cp_rms, cp_tail;
      cx_rms_class: cross cp_rms, cp_class;
    endgroup
    function new(string name, uvm_component parent);
      super.new(name, parent);
      stim_imp = new("stim_imp", this);
      cmd_imp = new("cmd_imp", this);
      read_imp = new("read_imp", this);
      write_imp = new("write_imp", this);
      resp_imp = new("resp_imp", this);
      contract_cov = new;
    endfunction
    function void write_stim(bb_blink_cmd_item item);
      if (!$cast(command, item.clone())) `uvm_fatal("STIM", "Cannot clone command")
      rows = layernorm_case_field(0);
      w = layernorm_case_field(1);
      lines = (w + 3) / 4;
      in_base = layernorm_case_field(2);
      out_base = layernorm_case_field(3);
      param_base = layernorm_case_field(4);
      rms = layernorm_case_field(5);
      for (int i = 0; i < 3; i++) begin
        bank[i]  = layernorm_case_field(6 + i);
        group[i] = layernorm_case_field(9 + i);
      end
      value_class = layernorm_case_field(12);
      backpressure = layernorm_case_field(13);
      writes = 0;
      responses = 0;
      // Port 0: x on every pass of a row (no sum pass with rms); port 1: eps
      // once, then per row gamma and beta of each bank row (gamma only with rms).
      addresses[0].delete();
      addresses[1].delete();
      addresses[1].push_back(param_base + (rms ? lines : 2 * lines));
      for (int r = 0; r < rows; r++) begin
        for (int p = 0; p < (rms ? 2 : 3); p++)
        for (int j = 0; j < lines; j++) addresses[0].push_back(in_base + r * lines + j);
        for (int j = 0; j < lines; j++) begin
          addresses[1].push_back(param_base + j);
          if (!rms) addresses[1].push_back(param_base + lines + j);
        end
      end
      mem_model.clear_mem();
      for (int r = 0; r < 1024; r++) begin
        mem_model.load_word_g(bank[0], group[0], r, fixture(0, r));
        mem_model.load_word_g(bank[1], group[1], r, fixture(1, r));
        mem_model.load_word_g(bank[2], group[2], r, {16{8'h5a}});
      end
      mem_model.arm();
      contract_cov.sample();
    endfunction
    function void write_cmd(bb_blink_cmd_item item);
      if (!item.compare(command)) `uvm_fatal("CMD", "Command fields differ")
      accepted = $time;
    endfunction
    function void write_read(bb_blink_read_item item);
      int p = item.port;
      int unsigned address;
      if (command == null || responses) `uvm_fatal("IDLE", "Bank read while no instruction runs")
      if (p < 0 || p >= 2 || item.rob_id != command.rob_id)
        `uvm_fatal("READ", "Invalid port or ROB")
      if (addresses[p].size() == 0) `uvm_fatal("READ", $sformatf("port%0d footprint exceeded", p))
      address = addresses[p].pop_front();
      if (item.bank_id != bank[p] || item.group_id != group[p] || item.addr != address)
        `uvm_fatal("READ", $sformatf(
                   "port%0d expected %0d/%0d/%0d got %0d/%0d/%0d",
                   p,
                   bank[p],
                   group[p],
                   address,
                   item.bank_id,
                   item.group_id,
                   item.addr
                   ))
      verification_pkg::evidence::checks++;
    endfunction
    function void write_write(bb_blink_write_item item);
      int unsigned row = out_base + writes;
      if (command == null || responses) `uvm_fatal("IDLE", "Bank write while no instruction runs")
      if (writes >= rows * lines || item.port != 0 || item.bank_id != bank[2] ||
          item.group_id != group[2] || item.rob_id != command.rob_id || item.addr != row ||
          item.mask != 16'hffff || item.data !== fixture(
              2, row
          ))
        `uvm_fatal("WRITE", $sformatf(
                   "Output mismatch/footprint: write%0d row%0d (expected row%0d) got %032h expected %032h",
                   writes,
                   item.addr,
                   row,
                   item.data,
                   fixture(
                       2, row
                   )
                   ))
      writes++;
      verification_pkg::evidence::checks++;
    endfunction
    function void write_resp(bb_blink_resp_item item);
      if (command == null || responses || item.rob_id != command.rob_id || writes != rows * lines ||
          addresses[0].size() || addresses[1].size())
        `uvm_fatal("RESP", "Invalid completion or missing reads/writes")
      for (int r = 0; r < 1024; r++) begin
        if (mem_model.mem[bank[0]][group[0]][r] !== fixture(
                0, r
            ) || mem_model.mem[bank[1]][group[1]][r] !== fixture(
                1, r
            ))
          `uvm_fatal("READONLY", $sformatf("Input or parameters changed at row%0d", r))
        if (mem_model.mem[bank[2]][group[2]][r] !== fixture(2, r))
          `uvm_fatal("GUARD", $sformatf("Output bank row%0d differs from the expected image", r))
      end
      responses++;
      verification_pkg::evidence::checks++;
      `uvm_info("LAYERNORM_TIMING", $sformatf(
                "rows%0d w%0d rms%0d class%0d cycles%0d",
                rows,
                w,
                rms,
                value_class,
                ($time - accepted) / 10
                ), UVM_LOW)
    endfunction
  endclass

  class layernorm_env extends bb_blink_env #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(layernorm_env)
    layernorm_scoreboard scb;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      scb = layernorm_scoreboard::type_id::create("scb", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      scb.mem_model = mem_model;
      cmd_agent.stim_ap.connect(scb.stim_imp);
      cmd_agent.cmd_ap.connect(scb.cmd_imp);
      read_mon.read_ap.connect(scb.read_imp);
      write_mon.write_ap.connect(scb.write_imp);
      resp_mon.resp_ap.connect(scb.resp_imp);
    endfunction
  endclass

  class layernorm_ball_test extends uvm_test;
    `uvm_component_utils(layernorm_ball_test)
    typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
    vif_t vif;
    layernorm_env env;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif)) `uvm_fatal("VIF", "Missing interface")
      bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(
          layernorm_cycles::get_type());
      env = layernorm_env::type_id::create("env", this);
    endfunction
    // One reset, then every case back to back: each instruction must restart
    // the Ball's state left by the previous one.
    task run_phase(uvm_phase phase);
      int selected, first, count;
      phase.raise_objection(this);
      if ($value$plusargs("LAYERNORM_CASE=%d", selected)) begin
        first = selected;
        count = 1;
      end else begin
        first = 0;
        count = layernorm_case_count();
      end
      vif.reset = 1;
      vif.cmd_resp_ready = 1;
      repeat (5) @(posedge vif.clock);
      vif.reset = 0;
      repeat (2) @(posedge vif.clock);
      for (int index = first; index < first + count; index++) begin
        layernorm_seq seq;
        int cycles;
        bit held;
        cycles = 0;
        layernorm_case_load(index);
        held = layernorm_case_field(13);
        if (held) begin
          @(negedge vif.clock);
          vif.cmd_resp_ready = 0;
        end
        seq = layernorm_seq::type_id::create("seq");
        seq.rob = index % (1 << `BB_ROB_ID_W);
        seq.start(env.cmd_agent.seqr);
        while (env.scb.responses == 0) begin
          @(posedge vif.clock);
          cycles++;
          if (held && vif.cmd_resp_valid) begin
            repeat (40) begin
              @(posedge vif.clock);
              if (!vif.cmd_resp_valid || vif.cmd_resp_bits_rob_id != seq.rob)
                `uvm_fatal("BACKPRESSURE", "Completion did not remain stable")
            end
            @(negedge vif.clock);
            vif.cmd_resp_ready = 1;
            held = 0;
          end
          if (cycles > 400000)
            `uvm_fatal("TIMEOUT", $sformatf(
                       "Case %0d did not complete: %0d writes, %0d/%0d reads left",
                       index,
                       env.scb.writes,
                       env.scb.addresses[0].size(),
                       env.scb.addresses[1].size()
                       ))
        end
        repeat (3) @(posedge vif.clock);
      end
      `uvm_info("LAYERNORM_COVERAGE", $sformatf("Contract coverage %0.2f",
                                                env.scb.contract_cov.get_inst_coverage()), UVM_LOW)
      phase.drop_objection(this);
    endtask
  endclass
  `include "tests/layernorm_throughput_test.svh"
  `include "tests/layernorm_latency_test.svh"
endpackage
