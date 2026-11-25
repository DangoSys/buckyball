package mxint5gemv_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function void mxint5gemv_bemu_row(
    input int unsigned role,
    input int unsigned row,
    input longint unsigned lo,
    input longint unsigned hi
  );
  import "DPI-C" function longint unsigned mxint5gemv_bemu_cycles(
    input longint unsigned iter,
    input longint unsigned rs2
  );
  import "DPI-C" function void mxint5_case_load(input int unsigned index);
  import "DPI-C" function int unsigned mxint5_case_field(input int unsigned field);
  import "DPI-C" function longint unsigned mxint5_case_word(
    input int unsigned role,
    row,
    high
  );

  function automatic bit [127:0] fixture(int role, row);
    return {mxint5_case_word(role, row, 1), mxint5_case_word(role, row, 0)};
  endfunction

  class mxint5gemv_seq extends uvm_sequence #(bb_blink_cmd_item);
    `uvm_object_utils(mxint5gemv_seq)
    function new(string name = "mxint5gemv_seq");
      super.new(name);
    endfunction
    task body();
      bb_blink_cmd_item item = bb_blink_cmd_item::type_id::create("item");
      start_item(item);
      item.rob_id = 7;
      item.funct = `MXINT5GEMV_FUNCT7;
      item.iter = mxint5_case_field(0);
      item.rs2 = 64'd2 | (64'd500 << 14) | (64'd3 << 28) | (64'd5 << 42) |
          (64'(mxint5_case_field(1)) << 56) | (64'(mxint5_case_field(2)) << 57);
      item.op1_bank = 1;
      item.op2_bank = 2;
      item.wr_bank = 3;
      item.op1_group = 1;
      item.op2_group = 2;
      item.wr_group = 1;
      item.op1_col = 1;
      item.op2_col = 4;
      item.wr_col = 1;
      item.wr_read_bank = 4;
      item.wr_arch_bank = 3;
      finish_item(item);
    endtask
  endclass

  class mxint5gemv_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(mxint5gemv_scoreboard)
    uvm_analysis_imp_stim #(bb_blink_cmd_item, mxint5gemv_scoreboard) stim_imp;
    uvm_analysis_imp_cmd #(bb_blink_cmd_item, mxint5gemv_scoreboard) cmd_imp;
    uvm_analysis_imp_read #(bb_blink_read_item, mxint5gemv_scoreboard) read_imp;
    uvm_analysis_imp_write #(bb_blink_write_item, mxint5gemv_scoreboard) write_imp;
    uvm_analysis_imp_resp #(bb_blink_resp_item, mxint5gemv_scoreboard) resp_imp;
    bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;
    bb_blink_cmd_item command;
    int writes, responses, reads[5], block_index[5], request_index[5];
    int k, blocks, code_rows, meta_rows, stride, accumulate, finalize;
    time accepted;
    covergroup contract_cov;
      cp_k: coverpoint k {
        bins short_k[] = {32, 64}; bins chunks[] = {512, 1024}; bins near_capacity = {2944};
      }
      cp_acc: coverpoint accumulate {bins off = {0}; bins on = {1};}
      cp_final: coverpoint finalize {bins off = {0}; bins on = {1};}
      modes: cross cp_acc, cp_final;
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
      k = item.iter;
      blocks = k / 32;
      code_rows = (k * 5 / 8 + 15) / 16;
      meta_rows = (k + 511) / 512;
      stride = code_rows + 2 * meta_rows;
      accumulate = item.rs2[56];
      finalize = item.rs2[57];
      writes = 0;
      responses = 0;
      foreach (reads[p]) begin
        reads[p] = 0;
        block_index[p] = 0;
        request_index[p] = 0;
      end
      mem_model.clear_mem();
      for (int row = 0; row < 1024; row++) begin
        mem_model.load_word_g(1, 1, row, fixture(0, row));
        for (int group = 0; group < 4; group++)
        mem_model.load_word_g(2, 2 + group, row, fixture(1 + group, row));
      end
      // Distinct physical versions for the old and new C states.
      for (int row = 0; row < 32; row++) begin
        mem_model.load_word_g(4, 1, row, fixture(5, row));
        mem_model.load_word_g(3, 1, row, {16{8'h5a}});
      end
      mem_model.arm();
      contract_cov.sample();
    endfunction
    function void write_cmd(bb_blink_cmd_item item);
      bb_blink_cmd_item expected = bb_blink_cmd_item::type_id::create("expected");
      expected.copy(command);
      if (!item.compare(expected)) `uvm_fatal("CMD", "Command fields differ")
      accepted = $time;
    endfunction
    function void write_read(bb_blink_read_item item);
      int p = item.port, addr, bank, group, b, index, code_count, count, col, part;
      if (p < 0 || p >= 5 || item.rob_id != 7) `uvm_fatal("READ", "Invalid port or ROB")
      if (p == 0 && reads[0] < 17 * accumulate) begin
        addr  = 5 + reads[0];
        bank  = 4;
        group = 1;
      end else begin
        b = block_index[p];
        index = request_index[p];
        if (b >= blocks) `uvm_fatal("READ", "Read after final block")
        if (p == 0) begin
          count = 2 + int'(b % 16 == 0);
          bank  = 1;
          group = 1;
          addr  = index < 2 ? 2 + b * 2 + index : 500 + b / 16;
        end else begin
          code_count = b % 4 == 0 ? 16 : 8;
          count = b % 16 == 0 ? 32 : code_count;
          bank = 2;
          group = 1 + p;
          col = index < code_count ? (b % 4 == 0 ? index / 2 : index) : index % 8;
          part = b % 4 == 0 ? index % 2 : 1;
          addr=3+col*stride+(index<code_count?(b*5)/4+part:
              code_rows+(index<24?0:meta_rows)+b/16);
        end
        request_index[p]++;
        if (request_index[p] == count) begin
          request_index[p] = 0;
          block_index[p]++;
        end
      end
      if (item.bank_id != bank || item.group_id != group || item.addr != addr)
        `uvm_fatal("READ", $sformatf(
                   "port%0d expected bank/group/row %0d/%0d/%0d got %0d/%0d/%0d",
                   p,
                   bank,
                   group,
                   addr,
                   item.bank_id,
                   item.group_id,
                   item.addr
                   ))
      reads[p]++;
      verification_pkg::evidence::checks++;
    endfunction
    function void write_write(bb_blink_write_item item);
      int count = finalize ? 3 : 17;
      if(writes>=count || item.port!=0 || item.bank_id!=3 || item.group_id!=1 || item.rob_id!=7
          || item.addr!=5+writes || item.mask!=16'hffff || item.data!==fixture(
              6, 5 + writes
          ))
        `uvm_fatal("WRITE", $sformatf(
                   "Invalid output/footprint at row%0d data%032h expected%032h",
                   item.addr,
                   item.data,
                   fixture(
                       6, 5 + writes
                   )
                   ))
      writes++;
      verification_pkg::evidence::checks++;
    endfunction
    function void write_resp(bb_blink_resp_item item);
      if (responses != 0 || item.rob_id != 7 || writes != (finalize ? 3 : 17))
        `uvm_fatal("RESP", "Invalid completion")
      foreach (block_index[p])
      if (block_index[p] != blocks || request_index[p] != 0) `uvm_fatal("RESP", "Missing reads")
      for (int row = 0; row < 32; row++) begin
        // Only the three finalized rows are defined; all other writes are forbidden.
        if ((row < 5 || row >= 5 + (finalize ? 3 : 17)) && mem_model.mem[3][1][row] !== {16{8'h5a}})
          `uvm_fatal("GUARD", "Destination guard changed")
      end
      responses++;
      verification_pkg::evidence::checks++;
      `uvm_info("INT5_TIMING", $sformatf(
                "K%0d acc%0d final%0d handshake_to_response_cycles%0d reads=%p",
                k,
                accumulate,
                finalize,
                ($time - accepted) / 10,
                reads
                ), UVM_LOW)
    endfunction
  endclass

  // MXINT5GEMV's cycles: its BEMU cycle model (emu/src/latency.rs), which reads
  // the A bank and the four B group banks for blocks of only zero products.
  // The instruction writes neither, so the bank model still holds them when
  // the response is valid.
  class mxint5gemv_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(mxint5gemv_cycles)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function longint unsigned expected(bb_blink_cmd_item cmd);
      bb_blink_env #(`BB_IN_BW, `BB_OUT_BW) env;
      bit [127:0] data;
      if (!$cast(env, get_parent())) `uvm_fatal("CYCLES", "parent is not a bb_blink_env")
      for (int unsigned row = 0; row < 1024; row++) begin
        data = env.mem_model.mem[cmd.op1_bank][cmd.op1_group][row];
        mxint5gemv_bemu_row(0, row, data[63:0], data[127:64]);
        for (int unsigned g = 0; g < 4; g++) begin
          data = env.mem_model.mem[cmd.op2_bank][cmd.op2_group+g][row];
          mxint5gemv_bemu_row(1 + g, row, data[63:0], data[127:64]);
        end
      end
      return mxint5gemv_bemu_cycles(cmd.iter, cmd.rs2);
    endfunction
  endclass

  class mxint5gemv_env extends bb_blink_env #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(mxint5gemv_env)
    mxint5gemv_scoreboard scb;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      scb = mxint5gemv_scoreboard::type_id::create("scb", this);
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

  class mxint5gemv_ball_test extends uvm_test;
    `uvm_component_utils(mxint5gemv_ball_test)
    typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
    vif_t vif;
    mxint5gemv_env env;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif)) `uvm_fatal("VIF", "Missing interface")
      bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(
          mxint5gemv_cycles::get_type());
      env = mxint5gemv_env::type_id::create("env", this);
    endfunction
    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      for (int index = 0; index < 10; index++) begin
        mxint5gemv_seq seq;
        int cycles;
        cycles = 0;
        vif.reset = 1;
        repeat (5) @(posedge vif.clock);
        vif.reset = 0;
        repeat (2) @(posedge vif.clock);
        if (index == 9) begin
          @(negedge vif.clock);
          vif.cmd_resp_ready = 0;
        end
        mxint5_case_load(index);
        seq = mxint5gemv_seq::type_id::create("seq");
        seq.start(env.cmd_agent.seqr);
        while (env.scb.responses == 0) begin
          @(posedge vif.clock);
          cycles++;
          if (index == 9 && cycles == 500) begin
            @(negedge vif.clock);
            vif.cmd_resp_ready = 1;
          end
          if(index==9 && cycles>450 && cycles<500 && (!vif.cmd_resp_valid || vif.cmd_resp_bits_rob_id!=7))
            `uvm_fatal("BACKPRESSURE", "Completion did not remain valid under backpressure")
          if (cycles > 20000) `uvm_fatal("TIMEOUT", "INT5 command did not complete")
        end
        repeat (3) @(posedge vif.clock);
      end
      phase.drop_objection(this);
    endtask
  endclass
  `include "tests/mxint5gemv_throughput_test.svh"
  `include "tests/mxint5gemv_latency_test.svh"
endpackage
