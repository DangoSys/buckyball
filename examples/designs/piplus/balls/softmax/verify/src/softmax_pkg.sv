package softmax_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function int unsigned softmax_case_count();
  import "DPI-C" function void softmax_case_load(input int unsigned index);
  import "DPI-C" function int unsigned softmax_case_field(input int unsigned field);
  import "DPI-C" function longint unsigned softmax_case_word(
    input int unsigned role,
    row,
    high
  );
  import "DPI-C" function int unsigned softmax_bank_entries();
  import "DPI-C" function longint unsigned softmax_bemu_cycles(
    input longint unsigned iter,
    input longint unsigned rs2,
    input int unsigned words[4096]
  );
  `include "softmax_cycles.svh"

  // Case fields (casegen/src/cases.rs).
  localparam int RowsField = 0;
  localparam int WidthField = 1;
  localparam int InBaseField = 2;
  localparam int OutBaseField = 3;
  localparam int ScaleField = 4;
  localparam int EventField = 5;
  // The case that holds cmdResp.ready low while the completion is pending.
  localparam int BackpressureCase = 16;

  // Bank row `row` of the input (role 0) or expected output (role 1) image.
  function automatic bit [127:0] fixture(int role, int row);
    return {softmax_case_word(role, row, 1), softmax_case_word(role, row, 0)};
  endfunction

  class softmax_seq extends uvm_sequence #(bb_blink_cmd_item);
    `uvm_object_utils(softmax_seq)
    int index;
    function new(string name = "softmax_seq");
      super.new(name);
    endfunction
    task body();
      bb_blink_cmd_item item = bb_blink_cmd_item::type_id::create("item");
      start_item(item);
      item.rob_id = index % (1 << `BB_ROB_ID_W);
      item.funct = `SOFTMAX_FUNCT7;
      item.iter = softmax_case_field(RowsField);
      item.rs2 = 64'(softmax_case_field(WidthField)) | (64'(softmax_case_field(InBaseField)) << 16)
          | (64'(softmax_case_field(OutBaseField)) << 30) |
          (64'(softmax_case_field(ScaleField) & 'hff) << 44);
      item.op1_bank = 1 + index % 4;
      item.wr_bank = 5 + index % 4;
      item.op1_group = index % 8;
      item.wr_group = 8 + index % 8;
      item.op1_col = 1;
      item.wr_col = 1;
      // Fields SOFTMAX does not use.
      item.op2_bank = 13;
      item.op2_group = 21;
      item.op2_col = 3;
      item.wr_read_bank = 17;
      item.wr_arch_bank = 18;
      finish_item(item);
    endtask
  endclass

  class softmax_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(softmax_scoreboard)
    uvm_analysis_imp_stim #(bb_blink_cmd_item, softmax_scoreboard) stim_imp;
    uvm_analysis_imp_cmd #(bb_blink_cmd_item, softmax_scoreboard) cmd_imp;
    uvm_analysis_imp_read #(bb_blink_read_item, softmax_scoreboard) read_imp;
    uvm_analysis_imp_write #(bb_blink_write_item, softmax_scoreboard) write_imp;
    uvm_analysis_imp_resp #(bb_blink_resp_item, softmax_scoreboard) resp_imp;
    bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;
    bb_blink_cmd_item command;
    int entries, rows, w, lines, in_base, out_base, scale, reads, writes, responses;
    int unsigned events[7];
    time accepted;

    covergroup command_cov;
      option.per_instance = 1;
      cp_tail: coverpoint (w % 4) {bins aligned = {0}; bins tail[] = {1, 2, 3};}
      cp_width: coverpoint w {
        bins minimum = {16};
        bins short_rows = {[17 : 63]};
        bins medium_rows = {[64 : 511]};
        bins long_rows = {[512 : 2047]};
        bins maximum = {2048};
      }
      cp_rows: coverpoint rows {
        bins one = {1}; bins few = {[2 : 7]}; bins many = {[8 : 63]}; bins lots = {[64 : $]};
      }
      cp_scale: coverpoint scale {
        bins minimum = {-126};
        bins negative = {[-125 : -1]};
        bins zero = {0};
        bins positive = {[1 : 126]};
        bins maximum = {127};
      }
      cp_in_base: coverpoint (in_base != 0);
      cp_out_base: coverpoint (out_base != 0);
      cp_in_last_row: coverpoint (in_base + rows * lines == entries);
      cp_out_last_row: coverpoint (out_base + rows * lines == entries);
      cp_rob: coverpoint command.rob_id;
      cp_underflow: coverpoint (events[0] != 0);
      cp_neg_inf: coverpoint (events[1] != 0);
      cp_difference_overflow: coverpoint (events[2] != 0);
      cp_max_tie: coverpoint (events[3] != 0);
      cp_subnormal_x: coverpoint (events[4] != 0);
      cp_subnormal_y: coverpoint (events[5] != 0);
      cp_u_near_minus_126: coverpoint (events[6] != 0);
    endgroup

    // x: 0 -inf, 1 zero, 2 subnormal, 3 negative normal, 4 positive normal.
    // y: 0 zero, 1 subnormal, 2 normal below 1, 3 one.
    covergroup element_cov with function sample (int x, int y);
      option.per_instance = 1;
      cp_x: coverpoint x {bins classes[] = {[0 : 4]};}
      cp_y: coverpoint y {bins classes[] = {[0 : 3]};}
    endgroup

    // The lanes after w of a row's last bank row: 0 inf, 1 NaN, 2 -inf.
    covergroup ignored_cov with function sample (int lane);
      option.per_instance = 1;
      cp_lane: coverpoint lane {bins classes[] = {[0 : 2]};}
    endgroup

    function new(string name, uvm_component parent);
      super.new(name, parent);
      stim_imp = new("stim_imp", this);
      cmd_imp = new("cmd_imp", this);
      read_imp = new("read_imp", this);
      write_imp = new("write_imp", this);
      resp_imp = new("resp_imp", this);
      command_cov = new;
      element_cov = new;
      ignored_cov = new;
    endfunction

    function void write_stim(bb_blink_cmd_item item);
      if (!$cast(command, item.clone())) `uvm_fatal("STIM", "Cannot clone command")
      entries = softmax_bank_entries();
      rows = softmax_case_field(RowsField);
      w = softmax_case_field(WidthField);
      lines = (w + 3) / 4;
      in_base = softmax_case_field(InBaseField);
      out_base = softmax_case_field(OutBaseField);
      scale = int'(softmax_case_field(ScaleField));
      foreach (events[i]) events[i] = softmax_case_field(EventField + i);
      reads = 0;
      writes = 0;
      responses = 0;
      mem_model.clear_mem();
      for (int r = 0; r < entries; r++) begin
        mem_model.load_word_g(item.op1_bank, item.op1_group, r, fixture(0, r));
        mem_model.load_word_g(item.wr_bank, item.wr_group, r, {16{8'h5a}});
      end
      mem_model.arm();
    endfunction

    function void write_cmd(bb_blink_cmd_item item);
      if (!item.compare(command)) `uvm_fatal("CMD", "Command fields differ")
      accepted = $time;
    endfunction

    // Each row is read three times in order: the maximum, the sum, the outputs.
    function void write_read(bb_blink_read_item item);
      int address = in_base + reads / (3 * lines) * lines + reads % lines;
      if (reads >= 3 * rows * lines) `uvm_fatal("READ", "Input footprint exceeded")
      if (item.port != 0 || item.bank_id != command.op1_bank || item.group_id != command.op1_group
          || item.rob_id != command.rob_id || item.addr != address)
        `uvm_fatal("READ", $sformatf(
                   "Read %0d expected bank %0d group %0d row %0d got bank %0d group %0d row %0d",
                   reads,
                   command.op1_bank,
                   command.op1_group,
                   address,
                   item.bank_id,
                   item.group_id,
                   item.addr
                   ))
      reads++;
      verification_pkg::evidence::checks++;
    endfunction

    function void write_write(bb_blink_write_item item);
      int address = out_base + writes;
      if (writes >= rows * lines) `uvm_fatal("WRITE", "Output footprint exceeded")
      if (item.port != 0 || item.bank_id != command.wr_bank || item.group_id != command.wr_group
          || item.rob_id != command.rob_id || item.addr != address || item.mask != 16'hffff)
        `uvm_fatal("WRITE", $sformatf(
                   "Write %0d expected row %0d got row %0d mask %04h",
                   writes,
                   address,
                   item.addr,
                   item.mask
                   ))
      if (item.data !== fixture(1, address))
        `uvm_fatal("WRITE", $sformatf(
                   "Row %0d (softmax row %0d) got %032h expected %032h",
                   address,
                   writes / lines,
                   item.data,
                   fixture(
                       1, address
                   )
                   ))
      writes++;
      verification_pkg::evidence::comparisons++;
    endfunction

    function void write_resp(bb_blink_resp_item item);
      if (responses != 0 || item.rob_id != command.rob_id || reads != 3 * rows * lines
          || writes != rows * lines)
        `uvm_fatal("RESP", $sformatf(
                   "Completion rob %0d after %0d reads and %0d writes", item.rob_id, reads, writes))
      for (int r = 0; r < entries; r++) begin
        if (mem_model.mem[command.op1_bank][command.op1_group][r] !== fixture(0, r))
          `uvm_fatal("READONLY", $sformatf("Input row %0d changed", r))
        if (mem_model.mem[command.wr_bank][command.wr_group][r] !== fixture(1, r))
          `uvm_fatal("GUARD", $sformatf("Output row %0d differs from the expected image", r))
      end
      command_cov.sample();
      for (int r = 0; r < rows; r++) begin
        for (int i = 0; i < lines * 4; i++) begin
          int row = r * lines + i / 4;
          bit [31:0] x = fixture(0, in_base + row) >> (32 * (i % 4));
          bit [31:0] y = fixture(1, out_base + row) >> (32 * (i % 4));
          if (i >= w) ignored_cov.sample(x == 32'h7f800000 ? 0 : x == 32'hff800000 ? 2 : 1);
          else
            element_cov.sample(
                x == 32'hff800000 ? 0 : x[30:0] == 0 ? 1 : x[30:23] == 0 ? 2 : x[31] ? 3 : 4,
                y == 0 ? 0 : y[30:23] == 0 ? 1 : y == 32'h3f800000 ? 3 : 2);
        end
      end
      responses++;
      verification_pkg::evidence::checks++;
      `uvm_info("SOFTMAX_TIMING", $sformatf(
                "rows %0d w %0d scale %0d cycles %0d", rows, w, scale, ($time - accepted) / 10),
                UVM_LOW)
    endfunction
  endclass

  class softmax_env extends bb_blink_env #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(softmax_env)
    softmax_scoreboard scb;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      scb = softmax_scoreboard::type_id::create("scb", this);
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

  class softmax_ball_test extends uvm_test;
    `uvm_component_utils(softmax_ball_test)
    typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
    vif_t vif;
    softmax_env env;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif)) `uvm_fatal("VIF", "Missing interface")
      bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(
          softmax_cycles::get_type());
      env = softmax_env::type_id::create("env", this);
    endfunction

    // One reset, then every case back to back: each command must restart the
    // Ball's state by itself.
    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      vif.reset = 1;
      vif.cmd_resp_ready = 1;
      repeat (5) @(posedge vif.clock);
      vif.reset = 0;
      repeat (2) @(posedge vif.clock);
      for (int index = 0; index < softmax_case_count(); index++) begin
        softmax_seq seq;
        int cycles = 0, held = 0;
        if (index == BackpressureCase) begin
          @(negedge vif.clock);
          vif.cmd_resp_ready = 0;
        end
        softmax_case_load(index);
        seq = softmax_seq::type_id::create("seq");
        seq.index = index;
        seq.start(env.cmd_agent.seqr);
        while (env.scb.responses == 0) begin
          @(posedge vif.clock);
          cycles++;
          if (cycles > 200000) `uvm_fatal("TIMEOUT", $sformatf("Case %0d did not complete", index))
          if (index == BackpressureCase && vif.cmd_resp_valid) begin
            if (vif.cmd_resp_bits_rob_id != index % (1 << `BB_ROB_ID_W) || vif.cmd_req_ready)
              `uvm_fatal("BACKPRESSURE", "Pending completion changed or a command was accepted")
            held++;
            if (held == 40) begin
              @(negedge vif.clock);
              vif.cmd_resp_ready = 1;
            end
          end
        end
        if (index == BackpressureCase && held < 40)
          `uvm_fatal("BACKPRESSURE", "Completion was not held")
        @(posedge vif.clock);
        if (vif.cmd_resp_valid || vif.status_running)
          `uvm_fatal("IDLE", $sformatf("Case %0d: the Ball is not idle after its completion", index
                     ))
      end
      `uvm_info("SOFTMAX_COVERAGE", $sformatf("command %0.2f element %0.2f ignored lanes %0.2f",
                                              env.scb.command_cov.get_inst_coverage(),
                                              env.scb.element_cov.get_inst_coverage(),
                                              env.scb.ignored_cov.get_inst_coverage()), UVM_LOW)
      phase.drop_objection(this);
    endtask
  endclass
  `include "tests/softmax_throughput_test.svh"
  `include "tests/softmax_latency_test.svh"
endpackage
