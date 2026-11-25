package mxint8gemv_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function void mxint8gemv_bemu_row(
    input int unsigned role,
    input int unsigned row,
    input longint unsigned lo,
    input longint unsigned hi
  );
  import "DPI-C" function longint unsigned mxint8gemv_bemu_cycles(
    input longint unsigned iter,
    input longint unsigned rs2
  );
  import "DPI-C" function int unsigned mxint8_case_count();
  import "DPI-C" function void mxint8_case_load(input int unsigned index);
  import "DPI-C" function int unsigned mxint8_case_field(input int unsigned field);
  import "DPI-C" function longint unsigned mxint8_case_word(
    input int unsigned role,
    row,
    high
  );

  // Roles: 0 A, 1..4 B groups 0..3, 5 old C, 6 new C before, 7 new C after.
  function automatic bit [127:0] fixture(int role, row);
    return {mxint8_case_word(role, row, 1), mxint8_case_word(role, row, 0)};
  endfunction

  class mxint8gemv_seq extends uvm_sequence #(bb_blink_cmd_item);
    `uvm_object_utils(mxint8gemv_seq)
    int rob;
    function new(string name = "mxint8gemv_seq");
      super.new(name);
    endfunction
    task body();
      bb_blink_cmd_item item = bb_blink_cmd_item::type_id::create("item");
      start_item(item);
      item.rob_id = rob;
      item.funct = `MXINT8GEMV_FUNCT7;
      item.iter = mxint8_case_field(0);
      item.rs2 = 64'(mxint8_case_field(1)) | (64'(mxint8_case_field(2)) << 14) |
          (64'(mxint8_case_field(3)) << 28) | (64'(mxint8_case_field(4)) << 42) |
          (64'(mxint8_case_field(5)) << 56) | (64'(mxint8_case_field(6)) << 57);
      item.op1_bank = mxint8_case_field(7);
      item.op2_bank = mxint8_case_field(8);
      item.wr_bank = mxint8_case_field(9);
      item.wr_read_bank = mxint8_case_field(10);
      item.wr_arch_bank = mxint8_case_field(9);
      item.op1_group = mxint8_case_field(11);
      item.op2_group = mxint8_case_field(12);
      item.wr_group = mxint8_case_field(13);
      item.op1_col = 1;
      item.op2_col = 4;
      item.wr_col = 1;
      finish_item(item);
    endtask
  endclass

  class mxint8gemv_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(mxint8gemv_scoreboard)
    uvm_analysis_imp_stim #(bb_blink_cmd_item, mxint8gemv_scoreboard) stim_imp;
    uvm_analysis_imp_cmd #(bb_blink_cmd_item, mxint8gemv_scoreboard) cmd_imp;
    uvm_analysis_imp_read #(bb_blink_read_item, mxint8gemv_scoreboard) read_imp;
    uvm_analysis_imp_write #(bb_blink_write_item, mxint8gemv_scoreboard) write_imp;
    uvm_analysis_imp_resp #(bb_blink_resp_item, mxint8gemv_scoreboard) resp_imp;
    bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;
    bb_blink_cmd_item command;
    int writes, responses, completed, old_reads, reads[5];
    int k, blocks, stride, a_codes, a_exp, b_base, c_base, accumulate, finalize;
    int op1_bank, op2_bank, wr_bank, wr_read_bank, op1_group, op2_group, wr_group;
    bit flags[18];
    time accepted;

    covergroup contract_cov;
      cp_k: coverpoint k {
        bins k32 = {32};
        bins below512 = {[64 : 480]};
        bins k512 = {512};
        bins below1024 = {[544 : 992]};
        bins k1024 = {1024};
        bins above1024 = {[1056 : 1952]};
        bins capacity = {1984};
      }
      cp_acc: coverpoint accumulate {bins off = {0}; bins on = {1};}
      cp_final: coverpoint finalize {bins off = {0}; bins on = {1};}
      modes: cross cp_acc, cp_final;
      cp_b_base: coverpoint b_base {bins zero = {0}; bins nonzero = {[1 : 1023]};}
      cp_c_base: coverpoint c_base {bins zero = {0}; bins mid = {[1 : 1006]}; bins top = {1007};}
      cp_a_top: coverpoint (a_exp + (blocks + 15) / 16 == 1024) {bins hit = {1};}
      cp_b_top: coverpoint (b_base + 8 * stride == 1024) {bins hit = {1};}
      cp_align: coverpoint flags[0] {bins hit = {1};}
      cp_align_flush: coverpoint flags[1] {bins hit = {1};}
      cp_term_round: coverpoint flags[2] {bins hit = {1};}
      cp_term_flush: coverpoint flags[3] {bins hit = {1};}
      cp_tie: coverpoint flags[4] {bins hit = {1};}
      cp_zero_block: coverpoint flags[5] {bins hit = {1};}
      cp_restart: coverpoint flags[6] {bins hit = {1};}
      cp_q_high: coverpoint flags[7] {bins hit = {1};}
      cp_q_low: coverpoint flags[8] {bins hit = {1};}
      cp_code_sat: coverpoint flags[9] {bins hit = {1};}
      cp_round_clamp: coverpoint flags[10] {bins hit = {1};}
      cp_left_shift: coverpoint flags[11] {bins hit = {1};}
      cp_zero_out: coverpoint flags[12] {bins hit = {1};}
      cp_edge_code: coverpoint flags[13] {bins hit = {1};}
      cp_k_limit: coverpoint flags[14] {bins hit = {1};}
      cp_old_valid: coverpoint flags[15] {bins hit = {1};}
      cp_old_zero: coverpoint flags[16] {bins hit = {1};}
      cp_inplace: coverpoint flags[17] {bins hit = {1};}
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
      k = mxint8_case_field(0);
      blocks = k / 32;
      stride = k / 16 + (k + 511) / 512;
      a_codes = mxint8_case_field(1);
      a_exp = mxint8_case_field(2);
      b_base = mxint8_case_field(3);
      c_base = mxint8_case_field(4);
      accumulate = mxint8_case_field(5);
      finalize = mxint8_case_field(6);
      op1_bank = mxint8_case_field(7);
      op2_bank = mxint8_case_field(8);
      wr_bank = mxint8_case_field(9);
      wr_read_bank = mxint8_case_field(10);
      op1_group = mxint8_case_field(11);
      op2_group = mxint8_case_field(12);
      wr_group = mxint8_case_field(13);
      foreach (flags[i]) flags[i] = mxint8_case_field(14 + i);
      writes = 0;
      responses = 0;
      old_reads = 0;
      foreach (reads[p]) reads[p] = 0;
      mem_model.clear_mem();
      // Old C is preloaded only when it is read; an in-place C is one image.
      for (int row = 0; row < 1024; row++) begin
        mem_model.load_word_g(op1_bank, op1_group, row, fixture(0, row));
        for (int g = 0; g < 4; g++)
        mem_model.load_word_g(op2_bank, op2_group + g, row, fixture(1 + g, row));
        if (accumulate) mem_model.load_word_g(wr_read_bank, wr_group, row, fixture(5, row));
        mem_model.load_word_g(wr_bank, wr_group, row, fixture(6, row));
      end
      mem_model.arm();
      contract_cov.sample();
    endfunction

    function void write_cmd(bb_blink_cmd_item item);
      if (!item.compare(command)) `uvm_fatal("CMD", "Command fields differ")
      accepted = $time;
    endfunction

    // Reads must stay inside the operand windows the contract assigns to each
    // channel: channel 0 A codes/exponents or old C, channel 1+g B group g.
    function void write_read(bb_blink_read_item item);
      int p = item.port, a = item.addr;
      bit ok;
      if (p < 0 || p >= 5 || item.rob_id != command.rob_id)
        `uvm_fatal("READ", $sformatf("Invalid port %0d or ROB %0d", p, item.rob_id))
      if (p == 0) begin
        if (accumulate && item.bank_id == wr_read_bank && item.group_id == wr_group
            && a >= c_base && a < c_base + 17) begin
          if (writes != 0) `uvm_fatal("READ", "Old C read after a C write")
          ok = 1;
          old_reads++;
        end else begin
          ok = item.bank_id == op1_bank && item.group_id == op1_group
              && (a >= a_codes && a < a_codes + k / 16
              || a >= a_exp && a < a_exp + (blocks + 15) / 16);
        end
      end else begin
        ok = item.bank_id == op2_bank && item.group_id == op2_group + p - 1
            && a >= b_base && a < b_base + 8 * stride;
      end
      if (!ok)
        `uvm_fatal("READ", $sformatf(
                   "port%0d read bank/group/row %0d/%0d/%0d outside its window",
                   p,
                   item.bank_id,
                   item.group_id,
                   a
                   ))
      reads[p]++;
      verification_pkg::evidence::checks++;
    endfunction

    function void write_write(bb_blink_write_item item);
      int count = finalize ? 3 : 17;
      if (writes >= count || item.port != 0 || item.bank_id != wr_bank || item.group_id != wr_group
          || item.rob_id != command.rob_id || item.addr != c_base + writes || item.mask != 16'hffff)
        `uvm_fatal("WRITE", $sformatf(
                   "Write %0d footprint: bank%0d group%0d row%0d mask%04h",
                   writes,
                   item.bank_id,
                   item.group_id,
                   item.addr,
                   item.mask
                   ))
      if (item.data !== fixture(7, item.addr))
        `uvm_fatal("WRITE", $sformatf(
                   "C row%0d (+%0d) got %032h expected %032h",
                   item.addr,
                   writes,
                   item.data,
                   fixture(
                       7, item.addr
                   )
                   ))
      writes++;
      verification_pkg::evidence::comparisons++;
    endfunction

    function void write_resp(bb_blink_resp_item item);
      if (responses != 0 || item.rob_id != command.rob_id || writes != (finalize ? 3 : 17))
        `uvm_fatal("RESP", $sformatf(
                   "Invalid completion rob%0d after %0d writes", item.rob_id, writes))
      if (old_reads != 17 * accumulate)
        `uvm_fatal("RESP", $sformatf("Old C read %0d rows, expected %0d", old_reads, 17 * accumulate
                   ))
      // The whole destination bank must equal the reference image, and no
      // source image may change.
      for (int row = 0; row < 1024; row++) begin
        if (mem_model.mem[wr_bank][wr_group][row] !== fixture(7, row))
          `uvm_fatal("GUARD", $sformatf("Destination row%0d differs from reference", row))
        if (mem_model.mem[op1_bank][op1_group][row] !== fixture(0, row))
          `uvm_fatal("READONLY", "A changed")
        for (int g = 0; g < 4; g++)
        if (mem_model.mem[op2_bank][op2_group+g][row] !== fixture(1 + g, row))
          `uvm_fatal("READONLY", "B changed")
        if (accumulate && wr_read_bank != wr_bank
            && mem_model.mem[wr_read_bank][wr_group][row] !== fixture(
                5, row
            ))
          `uvm_fatal("READONLY", "Old C changed")
      end
      responses++;
      completed++;
      verification_pkg::evidence::checks++;
      `uvm_info("MXINT8GEMV", $sformatf(
                "K%0d acc%0d final%0d cycles%0d reads=%p",
                k,
                accumulate,
                finalize,
                ($time - accepted) / 10,
                reads
                ), UVM_LOW)
    endfunction

    function void check_phase(uvm_phase phase);
      super.check_phase(phase);
      if (completed != mxint8_case_count())
        `uvm_fatal("CASES", $sformatf("%0d of %0d cases completed", completed, mxint8_case_count()))
      if (contract_cov.get_coverage() < 100.0)
        `uvm_fatal("COVERAGE", $sformatf("Contract coverage %0.2f%%", contract_cov.get_coverage()))
      `uvm_info("COVERAGE", $sformatf("Contract coverage %0.2f%%", contract_cov.get_coverage()),
                UVM_LOW)
    endfunction
  endclass

  // MXINT8GEMV's cycles: its BEMU cycle model (emu/src/latency.rs), which reads
  // the A bank and the four B group banks for blocks of only zero products.
  // The instruction writes neither, so the bank model still holds them when
  // the response is valid.
  class mxint8gemv_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(mxint8gemv_cycles)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function longint unsigned expected(bb_blink_cmd_item cmd);
      bb_blink_env #(`BB_IN_BW, `BB_OUT_BW) env;
      bit [127:0] data;
      if (!$cast(env, get_parent())) `uvm_fatal("CYCLES", "parent is not a bb_blink_env")
      for (int unsigned row = 0; row < 1024; row++) begin
        data = env.mem_model.mem[cmd.op1_bank][cmd.op1_group][row];
        mxint8gemv_bemu_row(0, row, data[63:0], data[127:64]);
        for (int unsigned g = 0; g < 4; g++) begin
          data = env.mem_model.mem[cmd.op2_bank][cmd.op2_group+g][row];
          mxint8gemv_bemu_row(1 + g, row, data[63:0], data[127:64]);
        end
      end
      return mxint8gemv_bemu_cycles(cmd.iter, cmd.rs2);
    endfunction
  endclass

  class mxint8gemv_env extends bb_blink_env #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(mxint8gemv_env)
    mxint8gemv_scoreboard scb;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      scb = mxint8gemv_scoreboard::type_id::create("scb", this);
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

  class mxint8gemv_ball_test extends uvm_test;
    `uvm_component_utils(mxint8gemv_ball_test)
    typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
    vif_t vif;
    mxint8gemv_env env;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif)) `uvm_fatal("VIF", "Missing interface")
      bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(
          mxint8gemv_cycles::get_type());
      env = mxint8gemv_env::type_id::create("env", this);
    endfunction
    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      for (int index = 0; index < mxint8_case_count(); index++) begin
        mxint8gemv_seq seq;
        int cycles, rob;
        cycles = 0;
        rob = index % (1 << `BB_ROB_ID_W);
        vif.reset = 1;
        repeat (5) @(posedge vif.clock);
        vif.reset = 0;
        repeat (2) @(posedge vif.clock);
        // Case 3 holds the completion under response backpressure.
        if (index == 3) begin
          @(negedge vif.clock);
          vif.cmd_resp_ready = 0;
        end
        mxint8_case_load(index);
        seq = mxint8gemv_seq::type_id::create("seq");
        seq.rob = rob;
        seq.start(env.cmd_agent.seqr);
        while (env.scb.responses == 0) begin
          @(posedge vif.clock);
          cycles++;
          if (index == 3 && cycles == 1500) begin
            @(negedge vif.clock);
            vif.cmd_resp_ready = 1;
          end
          if (index == 3 && cycles > 1400 && cycles < 1500
              && (!vif.cmd_resp_valid || vif.cmd_resp_bits_rob_id != rob))
            `uvm_fatal("BACKPRESSURE", "Completion did not remain valid under backpressure")
          if (cycles > 20000) `uvm_fatal("TIMEOUT", $sformatf("Case %0d did not complete", index))
        end
        repeat (3) @(posedge vif.clock);
      end
      phase.drop_objection(this);
    endtask
  endclass
  `include "tests/mxint8gemv_throughput_test.svh"
  `include "tests/mxint8gemv_latency_test.svh"
endpackage
