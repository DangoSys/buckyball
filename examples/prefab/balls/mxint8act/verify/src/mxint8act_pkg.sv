package mxint8act_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"

  import "DPI-C" function int unsigned mxint8act_num_cases();
  import "DPI-C" function int unsigned mxint8act_num_events();
  import "DPI-C" function int unsigned mxint8act_case_load(input int unsigned index);
  import "DPI-C" function int unsigned mxint8act_case_field(input int unsigned field);
  import "DPI-C" function longint unsigned mxint8act_case_word(
    input int unsigned role,
    row,
    high
  );
  import "DPI-C" function int unsigned mxint8act_case_events(input int unsigned kind);

  localparam int unsigned MXINT8ACT_BANK_ROWS = 1 << `BB_BANK_ADDR_W;
  // BEMU's cycle model (emu/src/latency.rs); `words` is the input bank, two
  // words per row, low half first.
  import "DPI-C" function longint unsigned mxint8act_bemu_cycles(
    input longint unsigned iter,
    rs2,
    input longint unsigned words[2*MXINT8ACT_BANK_ROWS],
    input int unsigned entries
  );

  // Semantic paths counted by casegen/src/reference.rs.
  localparam int unsigned MXINT8ACT_EVENTS = 19;
  localparam int unsigned MXINT8ACT_INTERIOR = 3;
  localparam int unsigned MXINT8ACT_TIMEOUT_CYCLES = 2000000;

  // Bank images: 0 input, 1 table, 2 output before, 3 expected output.
  function automatic bit [127:0] image(int unsigned role, int unsigned row);
    return {mxint8act_case_word(role, row, 1), mxint8act_case_word(role, row, 0)};
  endfunction

  // One MXINT8ACT command of the loaded case on the given banks.
  class mxint8act_seq extends uvm_sequence #(bb_blink_cmd_item);
    `uvm_object_utils(mxint8act_seq)
    bit [  `BB_ROB_ID_W-1:0] rob;
    bit [ `BB_BANK_ID_W-1:0] banks [3];
    bit [`BB_GROUP_ID_W-1:0] groups[3];

    function new(string name = "mxint8act_seq");
      super.new(name);
    endfunction

    task body();
      bb_blink_cmd_item item = bb_blink_cmd_item::type_id::create("item");
      start_item(item);
      item.rob_id = rob;
      item.funct = `MXINT8ACT_FUNCT7;
      item.iter = mxint8act_case_field(0);
      item.rs2 = 64'(mxint8act_case_field(2)) | (64'(mxint8act_case_field(1)) << 1) |
          (64'(mxint8act_case_field(3)) << 17) | (64'(mxint8act_case_field(4)) << 31) |
          (64'(mxint8act_case_field(5)) << 45);
      item.op1_bank = banks[0];
      item.op2_bank = banks[1];
      item.wr_bank = banks[2];
      item.op1_group = groups[0];
      item.op2_group = groups[1];
      item.wr_group = groups[2];
      item.op1_col = 1;
      item.op2_col = 1;
      item.wr_col = 1;
      item.wr_read_bank = banks[2];
      item.wr_arch_bank = banks[2];
      finish_item(item);
    endtask
  endclass

  // MXINT8ACT's cycles: its BEMU cycle model, on the input bank rows the
  // command reads (unchanged until its response).
  class mxint8act_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(mxint8act_cycles)
    bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function longint unsigned expected(bb_blink_cmd_item cmd);
      longint unsigned words[2*MXINT8ACT_BANK_ROWS];
      for (int unsigned r = 0; r < MXINT8ACT_BANK_ROWS; r++) begin
        words[2*r]   = mem_model.mem[cmd.op1_bank][cmd.op1_group][r][63:0];
        words[2*r+1] = mem_model.mem[cmd.op1_bank][cmd.op1_group][r][127:64];
      end
      return mxint8act_bemu_cycles(cmd.iter, cmd.rs2, words, MXINT8ACT_BANK_ROWS);
    endfunction
  endclass

  class mxint8act_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(mxint8act_scoreboard)
    uvm_analysis_imp_stim #(bb_blink_cmd_item, mxint8act_scoreboard) stim_imp;
    uvm_analysis_imp_cmd #(bb_blink_cmd_item, mxint8act_scoreboard) cmd_imp;
    uvm_analysis_imp_read #(bb_blink_read_item, mxint8act_scoreboard) read_imp;
    uvm_analysis_imp_write #(bb_blink_write_item, mxint8act_scoreboard) write_imp;
    uvm_analysis_imp_resp #(bb_blink_resp_item, mxint8act_scoreboard) resp_imp;
    bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;
    bb_blink_cmd_item command;
    int unsigned entries, rows, k, gated, in_base, out_base, table_base, stride, hold;
    int unsigned in_span, out_span, reads[2], writes, responses, kind;
    bit written[int unsigned];
    time accepted;

    covergroup shape_cg;
      cp_k: coverpoint k {
        bins one_block = {32};
        bins one_scale_row = {[64 : 480]};
        bins full_scale_row = {512};
        bins scale_rows = {[544 : 2048]};
        bins widest = {[2049 : $]};
      }
      cp_partial_scale_row: coverpoint (k % 512 != 0) {bins full = {0}; bins partial = {1};}
      cp_rows: coverpoint rows {
        bins one = {1}; bins few = {[2 : 16]}; bins many = {[17 : 64]}; bins most = {[65 : $]};
      }
      cp_gated: coverpoint gated {bins plain = {0}; bins gated = {1};}
      cp_input_top: coverpoint (in_base + in_span == entries) {bins below = {0}; bins top = {1};}
      cp_bases: coverpoint (in_base != 0) && (out_base != 0) {bins zero = {0}; bins offset = {1};}
      cp_hold: coverpoint (hold != 0) {bins ready = {0}; bins held = {1};}
      x_gated_rows: cross cp_gated, cp_rows;
      x_gated_k: cross cp_gated, cp_k;
    endgroup

    covergroup event_cg;
      cp_event: coverpoint kind {bins hit[] = {[0 : MXINT8ACT_EVENTS - 1]};}
    endgroup

    function new(string name, uvm_component parent);
      super.new(name, parent);
      stim_imp  = new("stim_imp", this);
      cmd_imp   = new("cmd_imp", this);
      read_imp  = new("read_imp", this);
      write_imp = new("write_imp", this);
      resp_imp  = new("resp_imp", this);
      shape_cg  = new;
      event_cg  = new;
    endfunction

    function void write_stim(bb_blink_cmd_item item);
      if (!$cast(command, item.clone())) `uvm_fatal("STIM", "Cannot clone command")
      rows = mxint8act_case_field(0);
      k = mxint8act_case_field(1);
      gated = mxint8act_case_field(2);
      in_base = mxint8act_case_field(3);
      out_base = mxint8act_case_field(4);
      table_base = mxint8act_case_field(5);
      stride = mxint8act_case_field(6);
      hold = mxint8act_case_field(7);
      in_span = rows * stride * (1 + gated);
      out_span = rows * stride;
      reads[0] = 0;
      reads[1] = 0;
      writes = 0;
      responses = 0;
      written.delete();
      mem_model.clear_mem();
      for (int unsigned r = 0; r < entries; r++) begin
        mem_model.load_word_g(command.op1_bank, command.op1_group, r, image(0, r));
        mem_model.load_word_g(command.op2_bank, command.op2_group, r, image(1, r));
        mem_model.load_word_g(command.wr_bank, command.wr_group, r, image(2, r));
      end
      mem_model.arm();
      shape_cg.sample();
      for (kind = 0; kind < MXINT8ACT_EVENTS; kind++)
      if (mxint8act_case_events(kind) != 0) event_cg.sample();
    endfunction

    function void write_cmd(bb_blink_cmd_item item);
      if (!item.compare(command)) `uvm_fatal("CMD", "Accepted command differs from the stimulus")
      accepted = $time;
    endfunction

    function void write_read(bb_blink_read_item item);
      if (item.rob_id != command.rob_id)
        `uvm_fatal("READ", $sformatf("port%0d ROB %0d", item.port, item.rob_id))
      if (item.port == 0) begin
        if (item.bank_id != command.op1_bank || item.group_id != command.op1_group ||
            item.addr < in_base || item.addr >= in_base + in_span)
          `uvm_fatal("READ", $sformatf(
                     "Input read bank%0d group%0d row%0d outside bank%0d group%0d rows [%0d,%0d)",
                     item.bank_id,
                     item.group_id,
                     item.addr,
                     command.op1_bank,
                     command.op1_group,
                     in_base,
                     in_base + in_span
                     ))
      end else begin
        if (item.port != 1 || item.bank_id != command.op2_bank || item.group_id != command.op2_group ||
            item.addr < table_base || item.addr >= table_base + 1024)
          `uvm_fatal("READ", $sformatf(
                     "Table read port%0d bank%0d group%0d row%0d outside bank%0d group%0d rows [%0d,%0d)",
                     item.port,
                     item.bank_id,
                     item.group_id,
                     item.addr,
                     command.op2_bank,
                     command.op2_group,
                     table_base,
                     table_base + 1024
                     ))
      end
      reads[item.port]++;
      verification_pkg::evidence::checks++;
    endfunction

    function void write_write(bb_blink_write_item item);
      if (item.port != 0 || item.bank_id != command.wr_bank || item.group_id != command.wr_group ||
          item.rob_id != command.rob_id || item.mask != 16'hffff)
        `uvm_fatal("WRITE", $sformatf(
                   "Write routing bank%0d group%0d ROB%0d mask%04h",
                   item.bank_id,
                   item.group_id,
                   item.rob_id,
                   item.mask
                   ))
      if (item.addr < out_base || item.addr >= out_base + out_span || written.exists(item.addr))
        `uvm_fatal("WRITE", $sformatf(
                   "Write row%0d outside rows [%0d,%0d) or written twice",
                   item.addr,
                   out_base,
                   out_base + out_span
                   ))
      if (item.data !== image(3, item.addr))
        `uvm_fatal("WRITE", $sformatf(
                   "Output row%0d (logical row%0d bank row%0d) got %032h expected %032h",
                   item.addr,
                   (item.addr - out_base) / stride,
                   (item.addr - out_base) % stride,
                   item.data,
                   image(
                       3, item.addr
                   )
                   ))
      written[item.addr] = 1;
      writes++;
      verification_pkg::evidence::comparisons++;
    endfunction

    function void write_resp(bb_blink_resp_item item);
      if (responses != 0 || item.rob_id != command.rob_id)
        `uvm_fatal("RESP", $sformatf(
                   "Completion ROB%0d expected one with ROB%0d", item.rob_id, command.rob_id))
      if (writes != out_span || reads[0] == 0 || (reads[1] == 0 && mxint8act_case_events(
              MXINT8ACT_INTERIOR
          ) != 0))
        `uvm_fatal("RESP", $sformatf(
                   "Completed after %0d of %0d writes, %0d input and %0d table reads",
                   writes,
                   out_span,
                   reads[0],
                   reads[1]
                   ))
      for (int unsigned r = 0; r < entries; r++) begin
        if (mem_model.mem[command.op1_bank][command.op1_group][r] !== image(0, r))
          `uvm_fatal("READONLY", $sformatf("Input row%0d changed", r))
        if (mem_model.mem[command.op2_bank][command.op2_group][r] !== image(1, r))
          `uvm_fatal("READONLY", $sformatf("Table row%0d changed", r))
        if (mem_model.mem[command.wr_bank][command.wr_group][r] !== image(3, r))
          `uvm_fatal("OUTPUT", $sformatf("Output bank row%0d differs from the reference", r))
      end
      responses++;
      verification_pkg::evidence::checks++;
      `uvm_info("MXINT8ACT", $sformatf(
                "M%0d K%0d gated%0d in%0d out%0d table%0d: %0d rows checked, %0d input and %0d table reads, %0d cycles",
                rows,
                k,
                gated,
                in_base,
                out_base,
                table_base,
                writes,
                reads[0],
                reads[1],
                ($time - accepted) / 10
                ), UVM_LOW)
    endfunction

    function void check_phase(uvm_phase phase);
      super.check_phase(phase);
      if (shape_cg.get_coverage() != 100.0)
        `uvm_fatal("COV", $sformatf("MXINT8ACT shape coverage %0.2f%%", shape_cg.get_coverage()))
      if (event_cg.get_coverage() != 100.0)
        `uvm_fatal("COV", $sformatf(
                   "MXINT8ACT semantic path coverage %0.2f%%", event_cg.get_coverage()))
      verification_pkg::evidence::checks++;
    endfunction
  endclass

  class mxint8act_env extends bb_blink_env #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(mxint8act_env)
    mxint8act_scoreboard scb;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      scb = mxint8act_scoreboard::type_id::create("scb", this);
    endfunction

    function void connect_phase(uvm_phase phase);
      mxint8act_cycles model;
      super.connect_phase(phase);
      $cast(model, cycles);
      model.mem_model = mem_model;
      scb.mem_model   = mem_model;
      cmd_agent.stim_ap.connect(scb.stim_imp);
      cmd_agent.cmd_ap.connect(scb.cmd_imp);
      read_mon.read_ap.connect(scb.read_imp);
      write_mon.write_ap.connect(scb.write_imp);
      resp_mon.resp_ap.connect(scb.resp_imp);
    endfunction
  endclass

  class mxint8act_ball_test extends uvm_test;
    `uvm_component_utils(mxint8act_ball_test)
    typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
    vif_t vif;
    mxint8act_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif)) `uvm_fatal("VIF", "Missing interface")
      bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(
          mxint8act_cycles::get_type());
      env = mxint8act_env::type_id::create("env", this);
    endfunction

    // Cases run back to back after one reset; each uses fresh banks, groups and ROB id.
    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      if (mxint8act_num_events() != MXINT8ACT_EVENTS)
        `uvm_fatal("EVENTS", $sformatf("casegen counts %0d semantic paths", mxint8act_num_events()))
      vif.reset = 1;
      repeat (5) @(posedge vif.clock);
      vif.reset = 0;
      repeat (2) @(posedge vif.clock);
      for (int unsigned index = 0; index < mxint8act_num_cases(); index++) run_case(index);
      phase.drop_objection(this);
    endtask

    task run_case(int unsigned index);
      mxint8act_seq seq = mxint8act_seq::type_id::create("seq");
      int unsigned cycles = 0, held = 0;
      env.scb.entries = mxint8act_case_load(index);
      seq.rob = $urandom;
      seq.banks[0] = $urandom;
      do seq.banks[1] = $urandom; while (seq.banks[1] == seq.banks[0]);
      do
        seq.banks[2] = $urandom;
      while (seq.banks[2] == seq.banks[0] || seq.banks[2] == seq.banks[1]);
      foreach (seq.groups[i]) seq.groups[i] = $urandom;
      if (mxint8act_case_field(7) != 0) begin
        @(negedge vif.clock);
        vif.cmd_resp_ready = 0;
      end
      seq.start(env.cmd_agent.seqr);
      while (env.scb.responses == 0) begin
        @(posedge vif.clock);
        cycles++;
        if (vif.cmd_resp_valid && !vif.cmd_resp_ready) begin
          if (vif.cmd_resp_bits_rob_id != seq.rob)
            `uvm_fatal("BACKPRESSURE", "Completion ROB id changed while held")
          held++;
          if (held == env.scb.hold) begin
            @(negedge vif.clock);
            vif.cmd_resp_ready = 1;
          end
        end else if (held != 0 && !vif.cmd_resp_ready)
          `uvm_fatal("BACKPRESSURE", "Completion dropped while held")
        if (cycles > MXINT8ACT_TIMEOUT_CYCLES)
          `uvm_fatal("TIMEOUT", $sformatf("MXINT8ACT case %0d timeout", index))
      end
      if (held != env.scb.hold)
        `uvm_fatal("BACKPRESSURE", $sformatf("Completion held %0d cycles", held))
      `uvm_info("MXINT8ACT", $sformatf("case %0d passed", index), UVM_LOW)
      repeat (3) @(posedge vif.clock);
    endtask
  endclass

  `include "tests/mxint8act_throughput_test.svh"
  `include "tests/mxint8act_latency_test.svh"
endpackage
