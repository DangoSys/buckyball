package mxint8matmul_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"

  localparam int MXINT8MATMUL_TIMEOUT_CYCLES = 500000;

  // One bank row: (bank, group, row) and its 128 bits, low word first.
  typedef struct {
    int unsigned bank;
    int unsigned group;
    int unsigned addr;
    int unsigned d0;
    int unsigned d1;
    int unsigned d2;
    int unsigned d3;
  } mxint8matmul_row_t;

  typedef struct {
    int unsigned iter;
    int unsigned rs2_lo;
    int unsigned rs2_hi;
    int unsigned op1_bank;
    int unsigned op2_bank;
    int unsigned wr_bank;
    int unsigned wr_read_bank;
    int unsigned op1_group;
    int unsigned op2_group;
    int unsigned wr_group;
    int unsigned features;
  } mxint8matmul_cmd_t;

  import "DPI-C" function int unsigned mxint8matmul_bank_entries();
  import "DPI-C" function int unsigned mxint8matmul_num_cases();
  import "DPI-C" function longint unsigned mxint8matmul_bemu_cycles(
    input longint unsigned iter,
    input longint unsigned rs2,
    input longint unsigned rows,
    input longint unsigned cols,
    input longint unsigned lanes
  );
  import "DPI-C" function int unsigned mxint8matmul_case_load(input int unsigned index);
  import "DPI-C" function int unsigned mxint8matmul_case_preloads();
  import "DPI-C" function void mxint8matmul_case_preload(
    input int unsigned i,
    output mxint8matmul_row_t row
  );
  import "DPI-C" function void mxint8matmul_case_cmd(
    input int unsigned step,
    output mxint8matmul_cmd_t cmd
  );
  import "DPI-C" function void mxint8matmul_case_expect(
    input int unsigned step,
    input int unsigned offset,
    output mxint8matmul_row_t row
  );

  // Every command of the loaded case, in order; ROB ids count up from `rob_base`.
  class mxint8matmul_seq extends uvm_sequence #(bb_blink_cmd_item);
    `uvm_object_utils(mxint8matmul_seq)

    int unsigned steps;
    int unsigned rob_base;

    function new(string name = "mxint8matmul_seq");
      super.new(name);
    endfunction

    task body();
      for (int unsigned step = 0; step < steps; step++) begin
        bb_blink_cmd_item  item = bb_blink_cmd_item::type_id::create($sformatf("item_%0d", step));
        mxint8matmul_cmd_t cmd;
        mxint8matmul_case_cmd(step, cmd);
        start_item(item);
        item.rob_id       = BB_BLINK_ROB_ID_W'(rob_base + step);
        item.funct        = `MXINT8MM_FUNCT7;
        item.rs2          = {cmd.rs2_hi, cmd.rs2_lo};
        item.iter         = cmd.iter;
        item.op1_bank     = cmd.op1_bank;
        item.op2_bank     = cmd.op2_bank;
        item.wr_bank      = cmd.wr_bank;
        item.op1_group    = cmd.op1_group;
        item.op2_group    = cmd.op2_group;
        item.wr_group     = cmd.wr_group;
        item.op1_col      = 1;
        item.op2_col      = 1;
        item.wr_col       = 1;
        item.wr_read_bank = cmd.wr_read_bank;
        item.wr_arch_bank = cmd.wr_bank;
        if (item.op1_bank != cmd.op1_bank || item.wr_read_bank != cmd.wr_read_bank ||
            item.op1_group != cmd.op1_group || item.op2_group != cmd.op2_group ||
            item.wr_group != cmd.wr_group)
          `uvm_fatal("CASE", "command field exceeds the configured BLink width")
        finish_item(item);
      end
    endtask
  endclass

  // Checks every bank access and the response of the live command against the
  // reference model: reads stay inside the A, B and (when accumulating) old C
  // windows, every C window row is written exactly once with the expected
  // value, and nothing outside it is written.
  class mxint8matmul_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(mxint8matmul_scoreboard)

    uvm_analysis_imp_stim #(bb_blink_cmd_item, mxint8matmul_scoreboard) stim_imp;
    uvm_analysis_imp_cmd #(bb_blink_cmd_item, mxint8matmul_scoreboard) cmd_imp;
    uvm_analysis_imp_read #(bb_blink_read_item, mxint8matmul_scoreboard) read_imp;
    uvm_analysis_imp_write #(bb_blink_write_item, mxint8matmul_scoreboard) write_imp;
    uvm_analysis_imp_resp #(bb_blink_resp_item, mxint8matmul_scoreboard) resp_imp;

    bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;
    bb_blink_cmd_item live[$];
    int unsigned stims, accepted, responses, total;
    // The accepted command, decoded from its fields.
    mxint8matmul_cmd_t cmd;
    int unsigned m, n, k, stride, a_base, b_base, c_base, c_rows;
    bit accumulate;
    bit written[];
    bit old_read[];
    time started;

    int m_shape, n_shape, at_capacity, any_base, renamed, rob;
    int feature;

    covergroup command_cg;
      cp_m: coverpoint m_shape {
        bins below_tile = {0}; bins one_tile = {1}; bins tiles_tail = {2}; bins tiles_whole = {3};
      }
      cp_n: coverpoint n_shape {
        bins below_tile = {0}; bins one_tile = {1}; bins tiles_tail = {2}; bins tiles_whole = {3};
      }
      cp_k: coverpoint k {
        bins one_block = {32};
        bins blocks = {[64 : 480]};
        bins one_exp_row = {512};
        bins two_exp_rows = {[544 : 992]};
        bins long = {[1024 : $]};
      }
      cp_acc: coverpoint accumulate {bins overwrite = {0}; bins accumulate = {1};}
      cp_old: coverpoint renamed iff (accumulate) {bins in_place = {0}; bins renamed = {1};}
      cp_capacity: coverpoint at_capacity {bins interior = {0}; bins last_row = {1};}
      cp_base: coverpoint any_base {bins zero = {0}; bins offset = {1};}
      cp_rob: coverpoint rob {bins id[] = {[0 : (1 << BB_BLINK_ROB_ID_W) - 1]};}
      x_acc_k: cross cp_acc, cp_k;
      x_m_n: cross cp_m, cp_n;
    endgroup

    // The semantic events of mxint8matmul_cmd_t.features (casegen/src/reference.rs).
    covergroup feature_cg;
      cp_feature: coverpoint feature {
        bins term_subnormal = {0};
        bins term_rounded = {1};
        bins term_tie_down = {2};
        bins term_tie_up = {3};
        bins term_neg_zero = {4};
        bins term_large = {5};
        bins product_extreme = {6};
        bins sum_inexact = {7};
        bins sum_cancel = {8};
        bins result_neg_zero = {9};
        bins result_subnormal = {10};
        bins code_min = {11};
        bins exp_zero = {12};
        bins exp_high = {13};
        bins old_special = {14};
      }
    endgroup

    function new(string name, uvm_component parent);
      super.new(name, parent);
      stim_imp   = new("stim_imp", this);
      cmd_imp    = new("cmd_imp", this);
      read_imp   = new("read_imp", this);
      write_imp  = new("write_imp", this);
      resp_imp   = new("resp_imp", this);
      command_cg = new();
      feature_cg = new();
    endfunction

    // Before a case: load its bank image and expect `steps` commands.
    function void start_case(int unsigned steps);
      mxint8matmul_row_t row;
      if (live.size() != 0) `uvm_fatal("SCB", "case started with a live command")
      mem_model.clear_mem();
      for (int unsigned i = 0; i < mxint8matmul_case_preloads(); i++) begin
        mxint8matmul_case_preload(i, row);
        mem_model.load_word_g(row.bank, row.group, row.addr, {row.d3, row.d2, row.d1, row.d0});
      end
      mem_model.arm();
      total     = steps;
      stims     = 0;
      accepted  = 0;
      responses = 0;
    endfunction

    function int shape(int unsigned size, int unsigned tile);
      if (size < tile) return 0;
      if (size == tile) return 1;
      return size % tile != 0 ? 2 : 3;
    endfunction

    function void write_stim(bb_blink_cmd_item item);
      bb_blink_cmd_item clone;
      if (stims >= total) `uvm_fatal("STIM", "more commands than the case holds")
      if (!$cast(clone, item.clone())) `uvm_fatal("STIM", "clone failed")
      live.push_back(clone);
      stims++;
    endfunction

    function void write_cmd(bb_blink_cmd_item item);
      if (live.size() == 0) `uvm_fatal("CMD", "command accepted without a stimulus")
      if (!item.compare(live[0]))
        `uvm_fatal("CMD", $sformatf(
                   "accepted command differs: got %s exp %s", item.sprint(), live[0].sprint()))
      mxint8matmul_case_cmd(accepted, cmd);
      m = item.rs2[9:0];
      n = item.rs2[19:10];
      a_base = item.rs2[33:20];
      b_base = item.rs2[47:34];
      c_base = item.rs2[61:48];
      accumulate = item.rs2[62];
      k = item.iter;
      stride = k / 16 + (k + 511) / 512;
      c_rows = m * n / 4;
      written = new[c_rows];
      old_read = new[c_rows];
      started = $time;
      m_shape = shape(m, `MXINT8MATMUL_ROWS);
      n_shape = shape(n, `MXINT8MATMUL_COLS);
      at_capacity = a_base + m * stride == mxint8matmul_bank_entries() ||
          b_base + n * stride == mxint8matmul_bank_entries() ||
          c_base + c_rows == mxint8matmul_bank_entries();
      any_base = a_base != 0 || b_base != 0 || c_base != 0;
      renamed = item.wr_read_bank != item.wr_bank;
      rob = item.rob_id;
      command_cg.sample();
      for (feature = 0; feature < 15; feature++) if (cmd.features[feature]) feature_cg.sample();
      accepted++;
      verification_pkg::evidence::checks++;
    endfunction

    function void write_read(bb_blink_read_item item);
      bb_blink_cmd_item c = current("READ");
      if (item.rob_id != c.rob_id) `uvm_fatal("READ", "read carries a foreign ROB id")
      if (item.port == 0 && item.bank_id == c.op1_bank && item.group_id == c.op1_group) begin
        if (item.addr < a_base || item.addr >= a_base + m * stride)
          `uvm_fatal("READ", $sformatf(
                     "A read of row %0d outside [%0d, %0d)", item.addr, a_base, a_base + m * stride
                     ))
      end else if (item.port == 0 && item.bank_id == c.wr_read_bank &&
                   item.group_id == c.wr_group) begin
        if (!accumulate) `uvm_fatal("READ", "old C read without accumulate")
        if (item.addr < c_base || item.addr >= c_base + c_rows)
          `uvm_fatal("READ", $sformatf(
                     "old C read of row %0d outside [%0d, %0d)", item.addr, c_base, c_base + c_rows
                     ))
        old_read[item.addr-c_base] = 1;
      end else if (item.port == 1 && item.bank_id == c.op2_bank && item.group_id == c.op2_group)
          begin
        if (item.addr < b_base || item.addr >= b_base + n * stride)
          `uvm_fatal("READ", $sformatf(
                     "B read of row %0d outside [%0d, %0d)", item.addr, b_base, b_base + n * stride
                     ))
      end else
        `uvm_fatal("READ", $sformatf(
                   "port %0d read of bank %0d group %0d is no operand",
                   item.port,
                   item.bank_id,
                   item.group_id
                   ))
      verification_pkg::evidence::checks++;
    endfunction

    function void write_write(bb_blink_write_item item);
      bb_blink_cmd_item c = current("WRITE");
      mxint8matmul_row_t want;
      int unsigned offset = item.addr - c_base;
      if (item.port != 0 || item.rob_id != c.rob_id || item.bank_id != c.wr_bank ||
          item.group_id != c.wr_group || item.mask != 16'hffff)
        `uvm_fatal("WRITE", $sformatf(
                   "write metadata port %0d rob %0d bank %0d group %0d mask %h",
                   item.port,
                   item.rob_id,
                   item.bank_id,
                   item.group_id,
                   item.mask
                   ))
      if (item.addr < c_base || offset >= c_rows)
        `uvm_fatal(
            "WRITE", $sformatf(
            "write of row %0d outside the C window [%0d, %0d)", item.addr, c_base, c_base + c_rows))
      if (written[offset]) `uvm_fatal("WRITE", $sformatf("C row %0d written twice", item.addr))
      mxint8matmul_case_expect(accepted - 1, offset, want);
      if (item.data !== {want.d3, want.d2, want.d1, want.d0})
        `uvm_fatal("DATA", $sformatf(
                   "step %0d M%0d N%0d K%0d C row %0d (elements %0d..%0d): got %032h exp %032h",
                   accepted - 1,
                   m,
                   n,
                   k,
                   item.addr,
                   offset * 4,
                   offset * 4 + 3,
                   item.data,
                   {
                     want.d3, want.d2, want.d1, want.d0
                   }
                   ))
      written[offset] = 1;
      verification_pkg::evidence::checks++;
    endfunction

    function void write_resp(bb_blink_resp_item item);
      bb_blink_cmd_item c = current("RESP");
      if (item.rob_id != c.rob_id) `uvm_fatal("RESP", "response carries a foreign ROB id")
      if (accepted != responses + 1) `uvm_fatal("RESP", "response without an accepted command")
      foreach (written[i])
      if (!written[i])
        `uvm_fatal("RESP", $sformatf("completed without writing C row %0d", c_base + i))
      if (accumulate)
        foreach (old_read[i])
        if (!old_read[i])
          `uvm_fatal("RESP", $sformatf("completed without reading old C row %0d", c_base + i))
      `uvm_info("MXINT8MM", $sformatf(
                "step %0d M%0d N%0d K%0d acc%0d rob%0d: %0d cycles",
                responses,
                m,
                n,
                k,
                accumulate,
                c.rob_id,
                ($time - started) / 10
                ), UVM_LOW)
      void'(live.pop_front());
      responses++;
      verification_pkg::evidence::checks++;
    endfunction

    function bb_blink_cmd_item current(string tag);
      if (live.size() == 0 || accepted != responses + 1)
        `uvm_fatal(tag, "bank access without a live command")
      return live[0];
    endfunction

    function void check_phase(uvm_phase phase);
      super.check_phase(phase);
      if (live.size() != 0) `uvm_fatal("SCB", "commands left without a response")
      if (command_cg.get_coverage() != 100.0)
        `uvm_fatal("COV", $sformatf("command coverage %0.2f%%", command_cg.get_coverage()))
      if (feature_cg.get_coverage() != 100.0)
        `uvm_fatal("COV", $sformatf("feature coverage %0.2f%%", feature_cg.get_coverage()))
    endfunction
  endclass

  // MXINT8MM's cycles: its BEMU cycle model (emu/src/latency.rs).
  class mxint8matmul_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(mxint8matmul_cycles)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function longint unsigned expected(bb_blink_cmd_item cmd);
      return mxint8matmul_bemu_cycles(cmd.iter, cmd.rs2, `MXINT8MATMUL_ROWS, `MXINT8MATMUL_COLS,
                                      `MXINT8MATMUL_LANES);
    endfunction
  endclass

  class mxint8matmul_env extends bb_blink_env #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(mxint8matmul_env)

    mxint8matmul_scoreboard scb;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      scb = mxint8matmul_scoreboard::type_id::create("scb", this);
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

  // Runs every case after a reset. Case `index` holds each response back for
  // (index % 3) * 20 cycles; while held, the response must stay stable and the
  // Ball must not take a new command.
  class mxint8matmul_ball_test extends uvm_test;
    `uvm_component_utils(mxint8matmul_ball_test)

    typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
    vif_t vif;
    mxint8matmul_env env;
    int unsigned issued;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
        `uvm_fatal("NOVIF", "bb_blink_if not found")
      bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(
          mxint8matmul_cycles::get_type());
      env = mxint8matmul_env::type_id::create("env", this);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      for (int unsigned index = 0; index < mxint8matmul_num_cases(); index++) run_case(index);
      phase.drop_objection(this);
    endtask

    task run_case(int unsigned index);
      mxint8matmul_seq seq = mxint8matmul_seq::type_id::create("seq");
      int unsigned hold = (index % 3) * 20;
      int unsigned held = 0;
      int unsigned cycles = 0;
      bit [BB_BLINK_ROB_ID_W-1:0] held_rob;
      vif.reset = 1'b1;
      repeat (5) @(posedge vif.clock);
      vif.reset = 1'b0;
      repeat (2) @(posedge vif.clock);
      vif.cmd_resp_ready = hold == 0;
      seq.steps = mxint8matmul_case_load(index);
      seq.rob_base = issued;
      issued += seq.steps;
      env.scb.start_case(seq.steps);
      fork
        seq.start(env.cmd_agent.seqr);
      join_none
      while (env.scb.responses != seq.steps) begin
        @(posedge vif.clock);
        cycles++;
        if (cycles > MXINT8MATMUL_TIMEOUT_CYCLES)
          `uvm_fatal("TIMEOUT", $sformatf(
                     "case %0d: %0d of %0d commands done", index, env.scb.responses, seq.steps))
        if (vif.cmd_resp_valid && !vif.cmd_resp_ready) begin
          if (held == 0) held_rob = vif.cmd_resp_bits_rob_id;
          if (vif.cmd_resp_bits_rob_id != held_rob || vif.cmd_req_ready)
            `uvm_fatal("BACKPRESSURE", "held response changed or the Ball took a new command")
          held++;
          if (held == hold) begin
            @(negedge vif.clock);
            vif.cmd_resp_ready = 1'b1;
          end
        end else if (vif.cmd_resp_valid && vif.cmd_resp_ready && hold != 0) begin
          held = 0;
          @(negedge vif.clock);
          vif.cmd_resp_ready = 1'b0;
        end
      end
      `uvm_info("MXINT8MM", $sformatf("case %0d: %0d commands passed in %0d cycles", index,
                                      seq.steps, cycles), UVM_LOW)
      repeat (3) @(posedge vif.clock);
    endtask
  endclass
  `include "tests/mxint8matmul_throughput_test.svh"
  `include "tests/mxint8matmul_latency_test.svh"
endpackage
