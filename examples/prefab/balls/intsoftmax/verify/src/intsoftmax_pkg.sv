package intsoftmax_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function longint unsigned intsoftmax_bemu_cycles(
    input longint unsigned iter,
    input longint unsigned rs2
  );
  import "DPI-C" function void intsoftmax_case_load(input int unsigned index);
  import "DPI-C" function int unsigned intsoftmax_case_field(input int unsigned field);
  import "DPI-C" function longint unsigned intsoftmax_case_word(
    input int unsigned role,
    row,
    high
  );
  import "DPI-C" function int unsigned intsoftmax_case_lookup(input int unsigned position);
  function automatic bit [127:0] fixture(int role, row);
    return {intsoftmax_case_word(role, row, 1), intsoftmax_case_word(role, row, 0)};
  endfunction

  class intsoftmax_seq extends uvm_sequence #(bb_blink_cmd_item);
    `uvm_object_utils(intsoftmax_seq)
    function new(string name = "intsoftmax_seq");
      super.new(name);
    endfunction
    task body();
      bb_blink_cmd_item item = bb_blink_cmd_item::type_id::create("item");
      start_item(item);
      item.rob_id = 7;
      item.funct = `INTSOFTMAX_FUNCT7;
      item.iter = intsoftmax_case_field(0);
      item.rs2 = 64'(intsoftmax_case_field(1)) | (64'(intsoftmax_case_field(3)) << 8);
      item.op1_bank = 1;
      item.op2_bank = 2;
      item.wr_bank = intsoftmax_case_field(2) ? 1 : 3;
      item.op1_group = 1;
      item.op2_group = 2;
      item.wr_group = 3;
      item.op1_col = 1;
      item.op2_col = 1;
      item.wr_col = 1;
      item.wr_read_bank = 3;
      item.wr_arch_bank = 3;
      finish_item(item);
    endtask
  endclass

  class intsoftmax_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(intsoftmax_scoreboard)
    uvm_analysis_imp_stim #(bb_blink_cmd_item, intsoftmax_scoreboard) stim_imp;
    uvm_analysis_imp_cmd #(bb_blink_cmd_item, intsoftmax_scoreboard) cmd_imp;
    uvm_analysis_imp_read #(bb_blink_read_item, intsoftmax_scoreboard) read_imp;
    uvm_analysis_imp_write #(bb_blink_write_item, intsoftmax_scoreboard) write_imp;
    uvm_analysis_imp_resp #(bb_blink_resp_item, intsoftmax_scoreboard) resp_imp;
    bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;
    bb_blink_cmd_item command;
    int writes, responses, reads[2], rows, k, stride;
    time accepted;
    covergroup contract_cov;
      cp_k: coverpoint k {bins selected[] = {1, 2, 3, 4, 6, 7, 32, 65, 127, 128};}
      cp_rows: coverpoint rows {bins selected[] = {1, 2, 3, 4, 8, 32, 1024};}
      cp_tail: coverpoint (k % 4) {bins aligned = {0}; bins tail[] = {1, 2, 3};}
      cp_capacity: coverpoint (rows * stride == 1024) {bins partial = {0}; bins full = {1};}
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
      rows = item.iter;
      k = item.rs2[7:0];
      stride = (k + 3) / 4;
      writes = 0;
      responses = 0;
      reads[0] = 0;
      reads[1] = 0;
      mem_model.clear_mem();
      for (int r = 0; r < 1024; r++) begin
        mem_model.load_word_g(1, 1, r, fixture(0, r));
        mem_model.load_word_g(2, 2, r, fixture(1, r));
        mem_model.load_word_g(3, 3, r, {16{8'h5a}});
      end
      mem_model.arm();
      if (!intsoftmax_case_field(4)) contract_cov.sample();
    endfunction
    function void write_cmd(bb_blink_cmd_item item);
      if (!item.compare(command)) `uvm_fatal("CMD", "Command fields differ")
      accepted = $time;
    endfunction
    function void write_read(bb_blink_read_item item);
      int p = item.port, bank, group, address;
      if (p < 0 || p >= 2 || item.rob_id != 7) `uvm_fatal("READ", "Invalid port or ROB")
      if (p == 0) begin
        if (reads[0] >= rows * stride) `uvm_fatal("READ", "Input footprint exceeded")
        bank = 1;
        group = 1;
        address = reads[0];
      end else begin
        if (reads[1] >= rows * k) `uvm_fatal("READ", "Table footprint exceeded")
        bank = 2;
        group = 2;
        address = intsoftmax_case_lookup(reads[1]) / 4;
      end
      if (item.bank_id != bank || item.group_id != group || item.addr != address)
        `uvm_fatal("READ", $sformatf(
                   "port%0d expected%0d/%0d/%0d got%0d/%0d/%0d",
                   p,
                   bank,
                   group,
                   address,
                   item.bank_id,
                   item.group_id,
                   item.addr
                   ))
      reads[p]++;
      verification_pkg::evidence::checks++;
    endfunction
    function void write_write(bb_blink_write_item item);
      if(writes>=rows*stride || item.port!=0 || item.bank_id!=3 || item.group_id!=3
          || item.rob_id!=7 || item.addr!=writes || item.mask!=16'hffff || item.data!==fixture(
              2, writes
          ))
        `uvm_fatal("WRITE", $sformatf(
                   "Output mismatch/footprint row%0d got%032h expected%032h",
                   item.addr,
                   item.data,
                   fixture(
                       2, writes
                   )
                   ))
      writes++;
      verification_pkg::evidence::checks++;
    endfunction
    function void write_resp(bb_blink_resp_item item);
      if (intsoftmax_case_field(4))
        `uvm_fatal("REJECT", "Invalid command completed instead of asserting")
      if(responses || item.rob_id!=7 || writes!=rows*stride || reads[0]!=rows*stride || reads[1]!=rows*k)
        `uvm_fatal("RESP", "Invalid completion or missing reads/writes")
      for (int r = 0; r < 1024; r++) begin
        if (mem_model.mem[1][1][r] !== fixture(0, r) || mem_model.mem[2][2][r] !== fixture(1, r))
          `uvm_fatal("READONLY", "Input/table changed")
        if (r >= rows * stride && mem_model.mem[3][3][r] !== {16{8'h5a}})
          `uvm_fatal("GUARD", "Output guard changed")
      end
      responses++;
      verification_pkg::evidence::checks++;
      `uvm_info("SOFTMAX_TIMING", $sformatf("M%0d K%0d cycles%0d", rows, k, ($time - accepted) / 10
                ), UVM_LOW)
    endfunction
  endclass

  // INTSOFTMAX's cycles: its BEMU cycle model (emu/src/latency.rs).
  class intsoftmax_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(intsoftmax_cycles)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function longint unsigned expected(bb_blink_cmd_item cmd);
      return intsoftmax_bemu_cycles(cmd.iter, cmd.rs2);
    endfunction
  endclass

  class intsoftmax_env extends bb_blink_env #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(intsoftmax_env)
    intsoftmax_scoreboard scb;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      scb = intsoftmax_scoreboard::type_id::create("scb", this);
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

  class intsoftmax_ball_test extends uvm_test;
    `uvm_component_utils(intsoftmax_ball_test)
    typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
    vif_t vif;
    intsoftmax_env env;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif)) `uvm_fatal("VIF", "Missing interface")
      bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(
          intsoftmax_cycles::get_type());
      env = intsoftmax_env::type_id::create("env", this);
    endfunction
    task run_phase(uvm_phase phase);
      int selected, count;
      phase.raise_objection(this);
      count = $value$plusargs("INTSOFTMAX_CASE=%d", selected) ? 1 : 12;
      for (int index = 0; index < count; index++) begin
        intsoftmax_seq seq;
        int cycles, which;
        cycles = 0;
        which = count == 1 ? selected : (index == 11 ? 19 : index);
        vif.reset = 1;
        vif.cmd_resp_ready = 1;
        repeat (5) @(posedge vif.clock);
        vif.reset = 0;
        repeat (2) @(posedge vif.clock);
        if (which == 9) begin
          @(negedge vif.clock);
          vif.cmd_resp_ready = 0;
        end
        intsoftmax_case_load(which);
        seq = intsoftmax_seq::type_id::create("seq");
        seq.start(env.cmd_agent.seqr);
        while (env.scb.responses == 0) begin
          @(posedge vif.clock);
          cycles++;
          if (which == 9 && cycles == 500) begin
            @(negedge vif.clock);
            vif.cmd_resp_ready = 1;
          end
          if(which==9 && cycles>200 && cycles<500 && (!vif.cmd_resp_valid || vif.cmd_resp_bits_rob_id!=7))
            `uvm_fatal("BACKPRESSURE", "Completion did not remain stable")
          if (cycles > 300000) `uvm_fatal("TIMEOUT", "Softmax command did not complete")
        end
        repeat (3) @(posedge vif.clock);
      end
      phase.drop_objection(this);
    endtask
  endclass
  `include "tests/intsoftmax_throughput_test.svh"
  `include "tests/intsoftmax_latency_test.svh"
endpackage
