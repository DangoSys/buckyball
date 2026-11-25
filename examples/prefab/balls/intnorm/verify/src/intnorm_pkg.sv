package intnorm_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"

  import "DPI-C" function int unsigned intnorm_case_count();
  import "DPI-C" function longint unsigned intnorm_bemu_cycles(
    input longint unsigned iter,
    input longint unsigned rs2
  );
  import "DPI-C" function void intnorm_case_load(input int unsigned index);
  import "DPI-C" function string intnorm_case_name();
  import "DPI-C" function longint unsigned intnorm_case_field(input int unsigned field);
  import "DPI-C" function longint unsigned intnorm_case_word(
    input int unsigned image,
    row,
    high
  );
  import "DPI-C" function longint unsigned intnorm_case_scratch(
    input int unsigned row,
    index,
    high
  );

  // Fields of the loaded case (casegen/src/cases.rs).
  typedef enum int unsigned {
    ROWS,
    K,
    INPUT_BASE,
    COEFFICIENT_BASE,
    OUTPUT_BASE,
    STALL,
    ENTRIES,
    BALL_ID,
    INPUT_TIES,
    CODE_TIES,
    CLIPS,
    ZERO_BLOCKS,
    NARROW_BLOCKS,
    FULL_BLOCKS,
    NEGATIVE_MEANS,
    EPSILON_ROWS,
    EPSILON_BITS
  } field_e;

  // Images of the loaded case.
  typedef enum int unsigned {
    INPUT,
    COEFFICIENTS,
    OUTPUT_BEFORE,
    OUTPUT_AFTER
  } image_e;

  function automatic int unsigned field(field_e f);
    return int'(intnorm_case_field(f));
  endfunction

  function automatic bit [127:0] image(image_e i, int unsigned row);
    return {intnorm_case_word(i, row, 1), intnorm_case_word(i, row, 0)};
  endfunction

  class intnorm_seq extends uvm_sequence #(bb_blink_cmd_item);
    `uvm_object_utils(intnorm_seq)
    bit [  BB_BLINK_ROB_ID_W-1:0] rob_id;
    bit [ BB_BLINK_BANK_ID_W-1:0] banks  [3];
    bit [BB_BLINK_GROUP_ID_W-1:0] groups [3];

    function new(string name = "intnorm_seq");
      super.new(name);
    endfunction

    task body();
      bb_blink_cmd_item item = bb_blink_cmd_item::type_id::create("item");
      start_item(item);
      item.rob_id = rob_id;
      item.funct = `INTNORM_FUNCT7;
      item.iter = field(ROWS);
      item.rs2 = 64'(field(K)) | 64'(field(INPUT_BASE)) << 16 | 64'(field(OUTPUT_BASE)) << 30 |
          64'(field(COEFFICIENT_BASE)) << 44;
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

  class intnorm_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(intnorm_scoreboard)
    uvm_analysis_imp_stim #(bb_blink_cmd_item, intnorm_scoreboard) stim_imp;
    uvm_analysis_imp_cmd #(bb_blink_cmd_item, intnorm_scoreboard) cmd_imp;
    uvm_analysis_imp_read #(bb_blink_read_item, intnorm_scoreboard) read_imp;
    uvm_analysis_imp_write #(bb_blink_write_item, intnorm_scoreboard) write_imp;
    uvm_analysis_imp_resp #(bb_blink_resp_item, intnorm_scoreboard) resp_imp;
    bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;
    bb_blink_cmd_item command;
    int unsigned entries, rows, k, stride, input_base, coefficient_base, output_base, scratch_base;
    int unsigned reads[2], scratch_writes, output_writes, responses;
    bit written[int unsigned];
    // Sampled values.
    int unsigned input_exponent, exponent_byte;
    bit input_sign;
    byte code;

    covergroup command_cov;
      cp_k: coverpoint k {bins selected[] = {512, 1024, 1536};}
      cp_rows: coverpoint rows {
        bins one = {1}; bins two = {2}; bins few = {[3 : 4]}; bins many = {[5 : 8]};
      }
      cp_k_rows: cross cp_k, cp_rows{
        ignore_bins too_many = binsof (cp_k) intersect {1024, 1536} && binsof (cp_rows.many);
        ignore_bins too_few = binsof (cp_k) intersect {1536} && binsof (cp_rows.few);
      }
      cp_input_end: coverpoint input_base + rows * k / 4 == entries;
      cp_coefficient_end: coverpoint coefficient_base + 1 + k / 2 == entries;
      cp_output_end: coverpoint scratch_base + k / 4 == entries;
      cp_input_base: coverpoint input_base {bins zero = {0}; bins other = {[1 : $]};}
      cp_coefficient_base: coverpoint coefficient_base {bins zero = {0}; bins other = {[1 : $]};}
      cp_output_base: coverpoint output_base {bins zero = {0}; bins other = {[1 : $]};}
      cp_epsilon: coverpoint field(
          EPSILON_BITS
      ) {
        bins one = {1};
        bins narrow = {[2 : 20]};
        bins moderate = {[21 : 44]};
        bins huge = {[45 : 64]};
      }
      cp_epsilon_dominates: coverpoint field(EPSILON_ROWS) != 0;
      cp_negative_mean: coverpoint field(NEGATIVE_MEANS) != 0;
      cp_input_ties: coverpoint field(INPUT_TIES) != 0;
      cp_code_ties: coverpoint field(CODE_TIES) != 0;
      cp_clips: coverpoint field(CLIPS) != 0;
      cp_zero_blocks: coverpoint field(ZERO_BLOCKS) != 0;
      cp_narrow_blocks: coverpoint field(NARROW_BLOCKS) != 0;
      cp_full_blocks: coverpoint field(FULL_BLOCKS) != 0;
      cp_stall: coverpoint field(STALL) != 0;
    endgroup

    // FP32 input classes of the Q16 conversion.
    covergroup input_cov;
      cp_exponent: coverpoint input_exponent {
        bins zero = {0};
        bins underflow = {[1 : 109]};
        bins half_step = {110};
        bins fraction = {[111 : 133]};
        bins whole = {[134 : 141]};
        bins limit = {142};
      }
      cp_sign: coverpoint input_sign;
      cp_class: cross cp_exponent, cp_sign{
        // +32768.0 does not fit signed Q16.
        ignore_bins positive_limit = binsof (cp_exponent.limit) && binsof (cp_sign) intersect {
          0
        };
      }
    endgroup

    // MXINT8 codes and block exponents written by the Ball.
    covergroup output_cov with function sample (bit is_code);
      cp_code: coverpoint code iff (is_code) {
        bins negative_limit = {-127};
        bins negative = {[-126 : -1]};
        bins zero = {0};
        bins positive = {[1 : 126]};
        bins positive_limit = {127};
        illegal_bins minimum = {-128};
      }
      cp_exponent: coverpoint exponent_byte iff (!is_code) {
        // 127 is a zero block or a block of 23 significant bits.
        bins unity = {127};
        bins narrow = {[105 : 110]};
        bins wide = {[111 : 126], [128 : 135]};
        bins full = {136};
        illegal_bins other = default;
      }
    endgroup

    function new(string name, uvm_component parent);
      super.new(name, parent);
      stim_imp = new("stim_imp", this);
      cmd_imp = new("cmd_imp", this);
      read_imp = new("read_imp", this);
      write_imp = new("write_imp", this);
      resp_imp = new("resp_imp", this);
      command_cov = new;
      input_cov = new;
      output_cov = new;
    endfunction

    function void write_stim(bb_blink_cmd_item item);
      if (!$cast(command, item.clone())) `uvm_fatal("STIM", "Cannot clone command")
      entries = field(ENTRIES);
      rows = field(ROWS);
      k = field(K);
      stride = k / 16 + k / 512;
      input_base = field(INPUT_BASE);
      coefficient_base = field(COEFFICIENT_BASE);
      output_base = field(OUTPUT_BASE);
      scratch_base = output_base + rows * stride;
      reads[0] = 0;
      reads[1] = 0;
      scratch_writes = 0;
      output_writes = 0;
      responses = 0;
      written.delete();
      mem_model.clear_mem();
      for (int unsigned r = 0; r < entries; r++) begin
        mem_model.load_word_g(item.op1_bank, item.op1_group, r, image(INPUT, r));
        mem_model.load_word_g(item.op2_bank, item.op2_group, r, image(COEFFICIENTS, r));
        mem_model.load_word_g(item.wr_bank, item.wr_group, r, image(OUTPUT_BEFORE, r));
      end
      mem_model.arm();
      command_cov.sample();
      for (int unsigned r = 0; r < rows * k / 4; r++) begin
        bit [127:0] word = image(INPUT, input_base + r);
        for (int i = 0; i < 4; i++) begin
          input_exponent = word[32*i+23+:8];
          input_sign = word[32*i+31];
          input_cov.sample();
        end
      end
    endfunction

    function void write_cmd(bb_blink_cmd_item item);
      if (!item.compare(command)) `uvm_fatal("CMD", "Accepted command differs from the driven one")
    endfunction

    function void write_read(bb_blink_read_item item);
      bit inside_input, inside_scratch, inside_coefficients;
      inside_input = item.bank_id == command.op1_bank && item.group_id == command.op1_group
          && item.addr >= input_base && item.addr < input_base + rows * k / 4;
      inside_scratch = item.bank_id == command.wr_bank && item.group_id == command.wr_group
          && item.addr >= scratch_base && item.addr < scratch_base + k / 4;
      inside_coefficients = item.bank_id == command.op2_bank && item.group_id == command.op2_group
          && item.addr >= coefficient_base && item.addr < coefficient_base + 1 + k / 2;
      if (item.rob_id != command.rob_id || item.port == 0 && !inside_input && !inside_scratch
          || item.port == 1 && !inside_coefficients || item.port > 1)
        `uvm_fatal("READ", $sformatf(
                   "port%0d bank%0d group%0d row%0d rob%0d is outside the footprint",
                   item.port,
                   item.bank_id,
                   item.group_id,
                   item.addr,
                   item.rob_id
                   ))
      reads[item.port]++;
      verification_pkg::evidence::checks++;
    endfunction

    function void write_write(bb_blink_write_item item);
      bit [127:0] expected;
      if (item.port != 0 || item.bank_id != command.wr_bank || item.group_id != command.wr_group
          || item.rob_id != command.rob_id || item.mask != 16'hffff)
        `uvm_fatal("WRITE", $sformatf(
                   "row%0d: bank%0d group%0d rob%0d mask%04h",
                   item.addr,
                   item.bank_id,
                   item.group_id,
                   item.rob_id,
                   item.mask
                   ))
      if (item.addr >= scratch_base && item.addr < scratch_base + k / 4) begin
        // Q16 results of one row, before they are packed to MXINT8.
        int unsigned row = scratch_writes / (k / 4), index = item.addr - scratch_base;
        if (row >= rows) `uvm_fatal("SCRATCH", "More scratch writes than rows")
        expected = {intnorm_case_scratch(row, index, 1), intnorm_case_scratch(row, index, 0)};
        if (item.data !== expected)
          `uvm_fatal("SCRATCH", $sformatf(
                     "row%0d elements %0d..%0d got %032h expected %032h",
                     row,
                     index * 4,
                     index * 4 + 3,
                     item.data,
                     expected
                     ))
        scratch_writes++;
      end else begin
        int unsigned offset = item.addr - output_base;
        if (item.addr < output_base || item.addr >= scratch_base || written.exists(item.addr))
          `uvm_fatal("WRITE", $sformatf("row%0d is outside the output or written twice", item.addr))
        expected = image(OUTPUT_AFTER, item.addr);
        if (item.data !== expected)
          `uvm_fatal("OUTPUT", $sformatf(
                     "output row%0d (row %0d, %s) got %032h expected %032h",
                     item.addr,
                     offset / stride,
                     offset % stride < k / 16 ? "codes" : "exponents",
                     item.data,
                     expected
                     ))
        written[item.addr] = 1;
        output_writes++;
        for (int i = 0; i < 16; i++) begin
          code = item.data[8*i+:8];
          exponent_byte = item.data[8*i+:8];
          output_cov.sample(offset % stride < k / 16);
        end
      end
      verification_pkg::evidence::checks++;
    endfunction

    function void write_resp(bb_blink_resp_item item);
      if (responses != 0 || item.rob_id != command.rob_id)
        `uvm_fatal("RESP", $sformatf("Completion %0d with rob%0d", responses, item.rob_id))
      if (scratch_writes != rows * k / 4 || output_writes != rows * stride)
        `uvm_fatal("RESP", $sformatf(
                   "Completed after %0d scratch and %0d output writes, expected %0d and %0d",
                   scratch_writes,
                   output_writes,
                   rows * k / 4,
                   rows * stride
                   ))
      for (int unsigned r = 0; r < entries; r++) begin
        if (mem_model.mem[command.op1_bank][command.op1_group][r] !== image(
                INPUT, r
            ) || mem_model.mem[command.op2_bank][command.op2_group][r] !== image(
                COEFFICIENTS, r
            ))
          `uvm_fatal("READONLY", $sformatf("Input or coefficient row%0d changed", r))
        if (mem_model.mem[command.wr_bank][command.wr_group][r] !== image(OUTPUT_AFTER, r))
          `uvm_fatal("IMAGE", $sformatf("Output bank row%0d differs from the reference", r))
      end
      responses++;
      verification_pkg::evidence::checks++;
    endfunction
  endclass

  // INTNORM's cycles: its BEMU cycle model (emu/src/latency.rs).
  class intnorm_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(intnorm_cycles)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function longint unsigned expected(bb_blink_cmd_item cmd);
      return intnorm_bemu_cycles(cmd.iter, cmd.rs2);
    endfunction
  endclass

  class intnorm_env extends bb_blink_env #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(intnorm_env)
    intnorm_scoreboard scb;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      scb = intnorm_scoreboard::type_id::create("scb", this);
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

  class intnorm_ball_test extends uvm_test;
    `uvm_component_utils(intnorm_ball_test)
    typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
    vif_t vif;
    intnorm_env env;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif)) `uvm_fatal("VIF", "Missing interface")
      bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(
          intnorm_cycles::get_type());
      env = intnorm_env::type_id::create("env", this);
    endfunction

    task run_phase(uvm_phase phase);
      int bid;
      phase.raise_objection(this);
      if (!$value$plusargs("BID=%d", bid)) `uvm_fatal("BID", "+BID=<n> is required")
      vif.reset = 1;
      repeat (5) @(posedge vif.clock);
      vif.reset = 0;
      repeat (2) @(posedge vif.clock);
      // Commands run back to back without a reset in between.
      for (int unsigned index = 0; index < intnorm_case_count(); index++) begin
        intnorm_seq seq = intnorm_seq::type_id::create("seq");
        int unsigned cycles, budget, stall, model;
        intnorm_case_load(index);
        if (bid != field(BALL_ID))
          `uvm_fatal("BID", $sformatf("+BID=%0d but the Ball id is %0d", bid, field(BALL_ID)))
        seq.rob_id = $urandom_range((1 << `BB_ROB_ID_W) - 1);
        for (int i = 0; i < 3; i++) begin
          seq.groups[i] = $urandom_range((1 << `BB_GROUP_ID_W) - 1);
          do
          seq.banks[i] = $urandom_range((1 << `BB_BANK_ID_W) - 1);
          while (i > 0 && seq.banks[i] == seq.banks[0] || i > 1 && seq.banks[i] == seq.banks[1]);
        end
        // The cycles of the BEMU model (emu/src/latency.rs).
        model  = intnorm_bemu_cycles(field(ROWS), field(K));
        budget = 2 * model + field(STALL);
        stall  = 0;
        cycles = 0;
        // Drive ready away from the sampling edge.
        @(negedge vif.clock);
        vif.cmd_resp_ready = field(STALL) == 0;
        seq.start(env.cmd_agent.seqr);
        while (env.scb.responses == 0) begin
          @(posedge vif.clock);
          cycles++;
          if (vif.cmd_resp_valid && !vif.cmd_resp_ready) begin
            if (vif.cmd_resp_bits_rob_id != seq.rob_id)
              `uvm_fatal("BACKPRESSURE", "Completion changed while stalled")
            stall++;
            if (stall == field(STALL)) begin
              @(negedge vif.clock);
              vif.cmd_resp_ready = 1;
            end
          end else if (stall != 0 && stall != field(STALL))
            `uvm_fatal("BACKPRESSURE", "Completion dropped while stalled")
          if (cycles > budget)
            `uvm_fatal("TIMEOUT", $sformatf("%s did not complete", intnorm_case_name()))
        end
        `uvm_info("INTNORM", $sformatf(
                  "case %0d %s rows%0d K%0d rob%0d banks %0d/%0d/%0d: passed in %0d cycles (model %0d)",
                  index,
                  intnorm_case_name(),
                  field(
                      ROWS
                  ),
                  field(
                      K
                  ),
                  seq.rob_id,
                  seq.banks[0],
                  seq.banks[1],
                  seq.banks[2],
                  cycles,
                  model
                  ), UVM_LOW)
        repeat (3) @(posedge vif.clock);
      end
      phase.drop_objection(this);
    endtask
  endclass
  `include "tests/intnorm_throughput_test.svh"
  `include "tests/intnorm_latency_test.svh"
endpackage
