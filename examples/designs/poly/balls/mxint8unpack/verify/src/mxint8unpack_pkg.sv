package mxint8unpack_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function void mxint8unpack_case_load(input int unsigned index);
  import "DPI-C" function int unsigned mxint8unpack_case_field(input int unsigned field);
  import "DPI-C" function longint unsigned mxint8unpack_case_word(
    input int unsigned role,
    row,
    high
  );
  // BEMU's cycle model (emu/src/latency.rs).
  import "DPI-C" function longint unsigned mxint8unpack_bemu_cycles(input longint unsigned iter);
  function automatic bit [127:0] fixture(int role, row);
    return {mxint8unpack_case_word(role, row, 1), mxint8unpack_case_word(role, row, 0)};
  endfunction

  class mxint8unpack_seq extends uvm_sequence #(bb_blink_cmd_item);
    `uvm_object_utils(mxint8unpack_seq)
    function new(string name = "mxint8unpack_seq");
      super.new(name);
    endfunction
    task body();
      bb_blink_cmd_item item = bb_blink_cmd_item::type_id::create("item");
      start_item(item);
      item.rob_id = 7;
      item.funct = `MXINT8UNPACK_FUNCT7;
      item.iter = mxint8unpack_case_field(0);
      item.rs2 = mxint8unpack_case_field(2);
      item.op1_bank = 1;
      item.op2_bank = item.iter == 1 ? 7 : 0;
      item.wr_bank = mxint8unpack_case_field(1) ? 1 : 3;
      item.op1_group = 1;
      item.op2_group = 0;
      item.wr_group = 3;
      item.op1_col = mxint8unpack_case_field(3);
      item.op2_col = 0;
      item.wr_col = mxint8unpack_case_field(4);
      item.wr_read_bank = 3;
      item.wr_arch_bank = 3;
      finish_item(item);
    endtask
  endclass

  // MXINT8UNPACK's cycles: its BEMU cycle model.
  class mxint8unpack_cycles extends bb_blink_cycles #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(mxint8unpack_cycles)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function longint unsigned expected(bb_blink_cmd_item cmd);
      return mxint8unpack_bemu_cycles(cmd.iter);
    endfunction
  endclass

  class mxint8unpack_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(mxint8unpack_scoreboard)
    uvm_analysis_imp_stim #(bb_blink_cmd_item, mxint8unpack_scoreboard) stim_imp;
    uvm_analysis_imp_cmd #(bb_blink_cmd_item, mxint8unpack_scoreboard) cmd_imp;
    uvm_analysis_imp_read #(bb_blink_read_item, mxint8unpack_scoreboard) read_imp;
    uvm_analysis_imp_write #(bb_blink_write_item, mxint8unpack_scoreboard) write_imp;
    uvm_analysis_imp_resp #(bb_blink_resp_item, mxint8unpack_scoreboard) resp_imp;
    bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;
    bb_blink_cmd_item command;
    int writes, responses, blocks, ignored_bank, data_class;
    int read_addresses[$];
    time accepted;
    covergroup contract_cov;
      option.per_instance = 1;
      cp_blocks: coverpoint blocks {bins selected[] = {1, 8, 16, 17, 33, 64, 127, 128};}
      cp_metadata: coverpoint ((blocks + 15) / 16) {bins selected[] = {1, 2, 3, 4, 8};}
      cp_ignored: coverpoint ignored_bank {bins selected[] = {0, 7};}
      cp_data: coverpoint data_class {bins selected[] = {0, 1, 2, 3};}
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
      blocks = item.iter;
      ignored_bank = item.op2_bank;
      data_class = blocks == 1 ? 0 : blocks == 8 ? 1 : blocks == 16 ? 2 : 3;
      writes = 0;
      responses = 0;
      read_addresses.delete();
      for (int b = 0; b < blocks; b++) begin
        if (b % 16 == 0) read_addresses.push_back(2 * blocks + b / 16);
        read_addresses.push_back(2 * b);
        read_addresses.push_back(2 * b + 1);
      end
      mem_model.clear_mem();
      for (int r = 0; r < 1024; r++) begin
        mem_model.load_word_g(1, 1, r, fixture(0, r));
        mem_model.load_word_g(3, 3, r, {16{8'h5a}});
      end
      mem_model.arm();
      if (!mxint8unpack_case_field(5)) contract_cov.sample();
    endfunction
    function void write_cmd(bb_blink_cmd_item item);
      if (!item.compare(command)) `uvm_fatal("CMD", "Command fields differ")
      accepted = $time;
    endfunction
    function void write_read(bb_blink_read_item item);
      int expected_address;
      if (read_addresses.size() == 0) `uvm_fatal("READ", "Input footprint exceeded")
      expected_address = read_addresses.pop_front();
      if(item.port!=0 || item.bank_id!=1 || item.group_id!=1 || item.rob_id!=7 || item.addr!=expected_address)
        `uvm_fatal("READ", $sformatf("Expected input row%0d got%0d", expected_address, item.addr))
      verification_pkg::evidence::checks++;
    endfunction
    function void write_write(bb_blink_write_item item);
      if(writes>=8*blocks || item.port!=0 || item.bank_id!=3 || item.group_id!=3 || item.rob_id!=7
          || item.addr!=writes || item.mask!=16'hffff || item.data!==fixture(
              1, writes
          ))
        `uvm_fatal("WRITE", $sformatf("Output mismatch/footprint row%0d", item.addr))
      writes++;
      verification_pkg::evidence::checks++;
    endfunction
    function void write_resp(bb_blink_resp_item item);
      if (mxint8unpack_case_field(5))
        `uvm_fatal("REJECT", "Invalid command completed instead of asserting")
      if (responses || item.rob_id != 7 || writes != 8 * blocks || read_addresses.size() != 0)
        `uvm_fatal("RESP", "Invalid completion or missing reads/writes")
      for (int r = 0; r < 1024; r++) begin
        if (mem_model.mem[1][1][r] !== fixture(0, r)) `uvm_fatal("READONLY", "Input changed")
        if (r >= 8 * blocks && mem_model.mem[3][3][r] !== {16{8'h5a}})
          `uvm_fatal("GUARD", "Output guard changed")
      end
      responses++;
      verification_pkg::evidence::checks++;
      `uvm_info("UNPACK_TIMING", $sformatf("B%0d cycles%0d", blocks, ($time - accepted) / 10),
                UVM_LOW)
    endfunction
  endclass

  class mxint8unpack_env extends bb_blink_env #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(mxint8unpack_env)
    mxint8unpack_scoreboard scb;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      scb = mxint8unpack_scoreboard::type_id::create("scb", this);
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

  class mxint8unpack_ball_test extends uvm_test;
    `uvm_component_utils(mxint8unpack_ball_test)
    typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
    vif_t vif;
    mxint8unpack_env env;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif)) `uvm_fatal("VIF", "Missing interface")
      bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(
          mxint8unpack_cycles::get_type());
      env = mxint8unpack_env::type_id::create("env", this);
    endfunction
    task run_phase(uvm_phase phase);
      int selected, count;
      phase.raise_objection(this);
      count = $value$plusargs("MXINT8UNPACK_CASE=%d", selected) ? 1 : 9;
      for (int index = 0; index < count; index++) begin
        mxint8unpack_seq seq;
        int cycles, which;
        cycles = 0;
        which = count == 1 ? selected : index;
        vif.reset = 1;
        vif.cmd_resp_ready = 1;
        repeat (5) @(posedge vif.clock);
        vif.reset = 0;
        repeat (2) @(posedge vif.clock);
        if (which == 8) begin
          @(negedge vif.clock);
          vif.cmd_resp_ready = 0;
        end
        mxint8unpack_case_load(which);
        seq = mxint8unpack_seq::type_id::create("seq");
        seq.start(env.cmd_agent.seqr);
        while (env.scb.responses == 0) begin
          @(posedge vif.clock);
          cycles++;
          if (which == 8 && cycles == 100) begin
            @(negedge vif.clock);
            vif.cmd_resp_ready = 1;
          end
          if(which==8 && cycles>60 && cycles<100 && (!vif.cmd_resp_valid || vif.cmd_resp_bits_rob_id!=7))
            `uvm_fatal("BACKPRESSURE", "Completion did not remain stable")
          if (cycles > 10000) `uvm_fatal("TIMEOUT", "Unpack command did not complete")
        end
        repeat (3) @(posedge vif.clock);
      end
      `uvm_info("UNPACK_COVERAGE", $sformatf("Contract coverage%0.2f",
                                             env.scb.contract_cov.get_inst_coverage()), UVM_LOW)
      phase.drop_objection(this);
    endtask
  endclass

  `include "tests/mxint8unpack_throughput_test.svh"
  `include "tests/mxint8unpack_latency_test.svh"
endpackage
