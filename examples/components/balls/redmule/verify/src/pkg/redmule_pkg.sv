package redmule_uvm_pkg;
  import uvm_pkg::*;
  import bb_uvm_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function void redmule_case_load(
    input int unsigned index,
    lanes
  );
  import "DPI-C" function longint unsigned redmule_case_shape();
  import "DPI-C" function longint unsigned redmule_case_word(
    input int unsigned bank,
    row,
    half
  );

  class redmule_mem_model extends bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(redmule_mem_model)
    int read_stalls[2], write_stalls, response_stalls;
    bit inject_stalls = 1;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    // Both performance ports model one-cycle memory; only functional cases inject stalls.
    task run_phase(uvm_phase phase);
      forever begin
        @(posedge vif.clock);
        if (vif.reset) begin
          for (int port = 0; port < 2; port++) begin
            vif.bank_read_req_ready[port]  <= 1;
            vif.bank_read_resp_valid[port] <= 0;
          end
          vif.bank_write_req_ready[0] <= 1;
          vif.bank_write_resp_valid[0] <= 0;
          vif.bank_write_resp_ok[0] <= 1;
          vif.cmd_resp_ready <= 1;
        end else begin
          for (int port = 0; port < 2; port++) begin
            if (vif.bank_read_resp_valid[port] && vif.bank_read_resp_ready[port])
              vif.bank_read_resp_valid[port] <= 0;
            if (vif.bank_read_req_valid[port] && vif.bank_read_req_ready[port]) begin
              if (!mem.exists(
                      vif.bank_read_bank_id[port]
                  ) || !mem[vif.bank_read_bank_id[port]].exists(
                      vif.bank_read_group_id[port]
                  ) || !mem[vif.bank_read_bank_id[port]][vif.bank_read_group_id[port]].exists(
                      vif.bank_read_req_addr[port]
                  ))
                `uvm_fatal("MEM", "bank read was not preloaded")
              vif.bank_read_resp_data[port] <= mem[vif.bank_read_bank_id[port]][vif.bank_read_group_id[port]][vif.bank_read_req_addr[port]];
              vif.bank_read_resp_valid[port] <= 1;
            end
            vif.bank_read_req_ready[port] <= (!vif.bank_read_resp_valid[port] || vif.bank_read_resp_ready[port]) &&
              (!inject_stalls || $urandom_range(
                0, 3
            ) != 0);
            if (vif.bank_read_req_valid[port] && !vif.bank_read_req_ready[port])
              read_stalls[port]++;
          end
          if (vif.bank_write_resp_valid[0] && vif.bank_write_resp_ready[0])
            vif.bank_write_resp_valid[0] <= 0;
          if (vif.bank_write_req_valid[0] && vif.bank_write_req_ready[0])
            vif.bank_write_resp_valid[0] <= 1;
          vif.bank_write_req_ready[0] <= (!vif.bank_write_resp_valid[0] || vif.bank_write_resp_ready[0]) &&
            (!inject_stalls || $urandom_range(
              0, 3
          ) != 0);
          vif.cmd_resp_ready <= (!inject_stalls || $urandom_range(0, 3) != 0);
          if (vif.bank_write_req_valid[0] && !vif.bank_write_req_ready[0]) write_stalls++;
          if (vif.cmd_resp_valid && !vif.cmd_resp_ready) response_stalls++;
        end
      end
    endtask
    function void check_phase(uvm_phase phase);
      verification_pkg::evidence::checks++;
      super.check_phase(phase);
      if (!read_stalls[0] || !read_stalls[1] || !write_stalls || !response_stalls)
        `uvm_fatal("STALL", "missing read/write/completion backpressure coverage")
    endfunction
  endclass

  class redmule_seq extends uvm_sequence #(bb_blink_cmd_item);
    `uvm_object_utils(redmule_seq)
    int rows, k, base, block_index, blocks;
    int groups[4];
    function new(string name = "redmule_seq");
      super.new(name);
    endfunction
    task body();
      bb_blink_cmd_item item = bb_blink_cmd_item::type_id::create("item");
      start_item(item);
      // bb_redmule(a=0, b=1, c=2, m, k, base, base, base, first): banks and K from rs1.
      // The frontend renames C on every write: block j writes bank 2 + j % 2 and a
      // continuation reads the previous C from the other one, its wr_read_bank.
      item.rob_id = block_index;
      item.funct = `REDMULE_F16_FUNCT7;
      item.rs2 = longint'(rows) | (longint'(base) << 12) |
        (longint'(base) << 22) | (longint'(base) << 32) |
        (longint'(block_index == 0) << 42);
      item.iter = k;
      item.op1_bank = 0;
      item.op2_bank = 1;
      item.wr_bank = 2 + block_index % 2;
      item.op1_group = groups[0];
      item.op2_group = groups[1];
      item.wr_group = groups[2];
      item.op1_col = 1;
      item.op2_col = 1;
      item.wr_col = 1;
      item.wr_read_bank = block_index == 0 ? 0 : 3 - block_index % 2;
      item.wr_arch_bank = 2;
      finish_item(item);
    endtask
  endclass

  class redmule_scoreboard extends uvm_scoreboard;
    `uvm_component_utils(redmule_scoreboard)
    uvm_analysis_imp_cmd #(bb_blink_cmd_item, redmule_scoreboard) cmd_imp;
    uvm_analysis_imp_read #(bb_blink_read_item, redmule_scoreboard) read_imp;
    uvm_analysis_imp_write #(bb_blink_write_item, redmule_scoreboard) write_imp;
    uvm_analysis_imp_resp #(bb_blink_resp_item, redmule_scoreboard) resp_imp;
    bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;
    int rows, k, base, blocks, block_index;
    int groups[4];
    int reads[2], writes, commands, responses;
    bit first;
    bit a_seen[int], b_seen[int], c_seen[int];
    int m_bucket, k_bucket, base_bucket, number_class;
    covergroup contract_cg;
      coverpoint first {bins start = {1}; bins continuation = {0};}
      coverpoint m_bucket {bins shapes[] = {[1 : 3]}; bins tile_rows[] = {16, 32};}
      coverpoint k_bucket {bins short_k = {8}; bins long_k = {16}; bins large_k = {32};}
      coverpoint base_bucket {bins zero = {0}; bins offset = {2};}
    endgroup
    covergroup number_cg;
      coverpoint number_class {bins classes[] = {[0 : 4]};}
    endgroup
    function new(string name, uvm_component parent);
      super.new(name, parent);
      cmd_imp = new("cmd_imp", this);
      read_imp = new("read_imp", this);
      write_imp = new("write_imp", this);
      resp_imp = new("resp_imp", this);
      contract_cg = new();
      number_cg = new();
    endfunction
    function void write_cmd(bb_blink_cmd_item item);
      if (item.funct != `REDMULE_F16_FUNCT7 || item.iter != k ||
          item.rs2[11:0] != rows || item.rob_id != block_index ||
          item.op1_bank != 0 || item.op2_bank != 1 || item.wr_bank != 2 + block_index % 2 ||
          item.op1_group != groups[0] || item.op2_group != groups[1] ||
          item.wr_group != groups[2] ||
          item.wr_read_bank != (block_index == 0 ? 0 : 3 - block_index % 2) ||
          item.wr_arch_bank != 2)
        `uvm_fatal("CMD", "unexpected RedMulE command")
      first = item.rs2[42];
      m_bucket = rows;
      k_bucket = k;
      base_bucket = base;
      contract_cg.sample();
      commands++;
    endfunction
    function void write_read(bb_blink_read_item item);
      int limit;
      if (item.rob_id != block_index) `uvm_fatal("READ", "unexpected read metadata")
      if (item.group_id != groups[item.bank_id]) `uvm_fatal("READ", "unexpected read group")
      if (item.port == 1 && item.bank_id == 1) begin
        limit = base + k;
        b_seen[item.addr] = 1;
      end else if (item.port == 0 && item.bank_id == 0) begin
        limit = base + rows * k / 8;
        a_seen[item.addr] = 1;
      end else if (item.port == 0 && item.bank_id == 3 - block_index % 2 && !first) begin
        limit = base + rows;
        c_seen[item.addr] = 1;
      end else `uvm_fatal("READ", "unexpected read bank")
      if (item.addr < base || item.addr >= limit) `uvm_fatal("READ", "read outside bank window")
      reads[item.port]++;
    endfunction
    function void write_write(bb_blink_write_item item);
      bit [127:0] expected;
      bit [ 15:0] value;
      if (writes >= rows) `uvm_fatal("WRITE", "unexpected early/extra write")
      expected = {
        redmule_case_word(2, block_index * rows + writes, 1),
        redmule_case_word(2, block_index * rows + writes, 0)
      };
      if (item.port != 0 || item.bank_id != 2 + block_index % 2 || item.addr != base + writes ||
          item.rob_id != block_index || item.group_id != groups[2] || item.mask != 16'hffff ||
          item.data !== expected)
        `uvm_fatal("WRITE", $sformatf(
                   "row %0d got %032h expected %032h", item.addr, item.data, expected))
      for (int lane = 0; lane < 8; lane++) begin
        value = item.data[lane*16+:16] & 16'h7fff;
        if (value == 0) number_class = 0;
        else if (value < 16'h0400) number_class = 1;
        else if (value < 16'h7c00) number_class = 2;
        else if (value == 16'h7c00) number_class = 3;
        else number_class = 4;
        number_cg.sample();
      end
      mem_model.load_word_g(item.bank_id, groups[2], item.addr, item.data);
      writes++;
    endfunction
    function void write_resp(bb_blink_resp_item item);
      if (item.rob_id != block_index || writes != rows)
        `uvm_fatal("RESP", "completion preceded required reads/writes")
      for (int row = base; row < base + rows * k / 8; row++)
      if (!a_seen.exists(row)) `uvm_fatal("READ", "missing A row")
      for (int row = base; row < base + k; row++)
      if (!b_seen.exists(row)) `uvm_fatal("READ", "missing B row")
      if (!first)
        for (int row = base; row < base + rows; row++)
        if (!c_seen.exists(row)) `uvm_fatal("READ", "missing C accumulator row")
      responses++;
    endfunction
    function void check_phase(uvm_phase phase);
      verification_pkg::evidence::checks++;
      super.check_phase(phase);
      if (commands != responses || commands == 0) `uvm_fatal("SCB", "missing command completion")
      if (contract_cg.get_coverage() != 100 || number_cg.get_coverage() != 100)
        `uvm_fatal(
            "COV", $sformatf(
            "contract %0.2f%% numbers %0.2f%%", contract_cg.get_coverage(), number_cg.get_coverage()
            ))
    endfunction
  endclass

  class redmule_env extends bb_blink_env #(`BB_IN_BW, `BB_OUT_BW);
    `uvm_component_utils(redmule_env)
    redmule_scoreboard scb;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      bb_blink_mem_model#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(
          redmule_mem_model::get_type());
      super.build_phase(phase);
      scb = redmule_scoreboard::type_id::create("scb", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      super.connect_phase(phase);
      scb.mem_model = mem_model;
      cmd_agent.cmd_ap.connect(scb.cmd_imp);
      read_mon.read_ap.connect(scb.read_imp);
      write_mon.write_ap.connect(scb.write_imp);
      resp_mon.resp_ap.connect(scb.resp_imp);
    endfunction
  endclass

  class redmule_ball_test extends uvm_test;
    `uvm_component_utils(redmule_ball_test)
    typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
    vif_t vif;
    redmule_env env;
    virtual redmule_reference_if reference;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
        `uvm_fatal("NOVIF", "bb_blink_if not found")
      if (!uvm_config_db#(virtual redmule_reference_if)::get(this, "", "reference", reference))
        `uvm_fatal("NOVIF", "native reference interface not found")
      env = redmule_env::type_id::create("env", this);
    endfunction
    task run_phase(uvm_phase phase);
      int rows, k, base, blocks, before_response, cycles;
      int groups[4];
      longint unsigned shape;
      longint native_cycles, ball_started, ball_cycles, total_native = 0, total_ball = 0;
      redmule_seq seq;
      redmule_mem_model memory_agent;
      if (!$cast(memory_agent, env.mem_model)) `uvm_fatal("MEM", "wrong memory model type")
      phase.raise_objection(this);
      for (int index = 0; index < 67; index++) begin
        vif.reset = 1;
        repeat (5) @(posedge vif.clock);
        vif.reset = 0;
        repeat (2) @(posedge vif.clock);
        memory_agent.inject_stalls = index < 64;
        redmule_case_load(index, `REDMULE_LANES);
        shape = redmule_case_shape();
        rows = shape[15:0];
        k = shape[31:16];
        base = shape[47:32];
        blocks = shape[63:48];
        env.scb.rows = rows;
        env.scb.k = k;
        env.scb.base = base;
        env.scb.blocks = blocks;
        // Odd cases place A, B and C at nonzero group bases of their banks; both
        // banks of C use the destination group.
        groups = index % 2 ? '{1, 2, 3, 3} : '{0, 0, 0, 0};
        env.scb.groups = groups;
        env.mem_model.clear_mem();
        for (int row = 0; row < base + rows * k / 8; row++) begin
          env.mem_model.load_word_g(0, groups[0], row, {
                                    redmule_case_word(0, row, 1), redmule_case_word(0, row, 0)});
          reference.load_word(0, row, {redmule_case_word(0, row, 1), redmule_case_word(0, row, 0)});
        end
        for (int row = 0; row < base + k; row++) begin
          env.mem_model.load_word_g(1, groups[1], row, {
                                    redmule_case_word(1, row, 1), redmule_case_word(1, row, 0)});
          reference.load_word(1, row, {redmule_case_word(1, row, 1), redmule_case_word(1, row, 0)});
        end
        for (int row = 0; row < base + rows; row++) begin
          env.mem_model.load_word_g(2, groups[2], row, 0);
          reference.load_word(2, row, 0);
        end
        env.mem_model.arm();
        for (int block_index = 0; block_index < blocks; block_index++) begin
          env.scb.block_index = block_index;
          env.scb.reads = '{0, 0};
          env.scb.writes = 0;
          env.scb.a_seen.delete();
          env.scb.b_seen.delete();
          env.scb.c_seen.delete();
          before_response = env.scb.responses;
          seq = redmule_seq::type_id::create("seq");
          seq.groups = groups;
          seq.rows = rows;
          seq.k = k;
          seq.base = base;
          seq.block_index = block_index;
          seq.blocks = blocks;
          reference.execute(rows, k, base, block_index == 0, native_cycles);
          for (int row = 0; row < rows; row++)
          if (reference.read_word(
                  base + row
              ) !== {redmule_case_word(
                  2, block_index * rows + row, 1
              ), redmule_case_word(
                  2, block_index * rows + row, 0
              )})
            `uvm_fatal("REFERENCE", "native engine disagrees with golden")
          ball_started = reference.get_ticks();
          seq.start(env.cmd_agent.seqr);
          cycles = 0;
          while (env.scb.responses == before_response) begin
            @(posedge vif.clock);
            if (++cycles > 200000) `uvm_fatal("TIMEOUT", "RedMulE command timed out")
          end
          ball_cycles = reference.get_ticks() - ball_started;
          total_native += native_cycles;
          total_ball += ball_cycles;
          `uvm_info("PERF", $sformatf(
                    "M=%0d K=%0d N=8 first=%0d native=%0d ball=%0d",
                    rows,
                    k,
                    block_index == 0,
                    native_cycles,
                    ball_cycles
                    ), UVM_LOW)
          repeat (2) @(posedge vif.clock);
        end
        `uvm_info("REDMULE", $sformatf("case %0d passed", index), UVM_LOW)
      end
      `uvm_info("PERF_TOTAL", $sformatf("native=%0d ball=%0d", total_native, total_ball), UVM_LOW)
      phase.drop_objection(this);
    endtask
  endclass
endpackage
