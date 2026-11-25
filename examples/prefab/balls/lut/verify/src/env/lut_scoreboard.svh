class lut_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(lut_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, lut_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, lut_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, lut_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, lut_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, lut_scoreboard) resp_imp;

  bb_blink_mem_model #(`BB_IN_BW, `BB_OUT_BW) mem_model;

  lut_cmd_item stim_q[$];
  int unsigned expected_reads[`BB_IN_BW];
  int unsigned expected_writes;
  int unsigned cmd_count;
  int unsigned read_count[`BB_IN_BW];
  int unsigned write_count;
  int unsigned resp_count;
  int signed maximum;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_imp  = new("stim_imp", this);
    cmd_imp   = new("cmd_imp", this);
    read_imp  = new("read_imp", this);
    write_imp = new("write_imp", this);
    resp_imp  = new("resp_imp", this);
  endfunction

  function void write_stim(bb_blink_cmd_item item);
    lut_cmd_item got;
    lut_cmd_item clone;
    int i;

    if (stim_q.size() != 0) begin
      `uvm_fatal("SCB", "single outstanding command supported")
    end
    if (!$cast(got, item)) begin
      `uvm_fatal("SCB", "stim item is not lut_cmd_item")
    end
    if (!$cast(clone, got.clone())) begin
      `uvm_fatal("SCB", "failed to clone stimulus item")
    end
    stim_q.push_back(clone);

    if (mem_model == null) begin
      `uvm_fatal("SCB", "mem_model handle not set")
    end
    mem_model.clear_mem();
    for (i = 0; i < clone.num_src_words; i++) begin
      mem_model.load_word_g(int'(clone.op1_bank), int'(clone.op1_group), i, clone.src_words[i]);
    end
    maximum = 32'sh80000000;
    for (i = 0; i < clone.num_src_words; i++)
    for (int lane = 0; lane < 4; lane++)
    if ($signed(clone.src_words[i][lane*32+:32]) > maximum)
      maximum = $signed(clone.src_words[i][lane*32+:32]);
    // Byte lane tables span four groups; word tables occupy one group.
    for (i = 0; i < clone.num_lut_words; i++) begin
      if (clone.rs2[1:0] == 0)
        mem_model.load_word_g(int'(clone.op2_bank), int'(clone.op2_group) + i / 64, i % 64,
                              clone.lut_words[i]);
      else
        mem_model.load_word_g(int'(clone.op2_bank), int'(clone.op2_group), i, clone.lut_words[i]);
    end
    mem_model.arm();

    expected_writes   = clone.num_dst_words * (clone.rs2[1:0] == 2 ? 2 : 1);
    expected_reads[0] = clone.iter * (clone.rs2[1:0] == 2 ? 3 : 1);
    expected_reads[1] = clone.iter * (clone.rs2[1:0] == 0 ? 16 : 4);
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    lut_cmd_item stim;
    stim = current_stim("CMD");
    check_cmd(item, stim);
    cmd_count++;
  endfunction

  function void check_cmd(bb_blink_cmd_item got, lut_cmd_item exp);
    // Compare as plain BLink commands; the comparer rejects differing types.
    bb_blink_cmd_item want = bb_blink_cmd_item::type_id::create("want");
    want.copy(exp);
    verification_pkg::evidence::checks++;
    if (!got.compare(want))
      `uvm_fatal("CMD", $sformatf("command mismatch: got %s exp %s", got.sprint(), want.sprint()))
  endfunction

  function void write_read(bb_blink_read_item item);
    lut_cmd_item stim;
    int unsigned expect_bank;
    int unsigned expect_addr;
    int unsigned expect_group;
    int unsigned row;
    int unsigned lane;
    bit [7:0] input_byte;
    int signed input_word;
    int unsigned index;
    longint signed distance;

    stim = current_stim("READ");
    if (item.port != 0 && item.port != 1)
      `uvm_fatal("READ", $sformatf("LutBall has two read ports, got port %0d", item.port))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))

    if (item.port == 0) begin
      expect_bank  = stim.rs2[1:0] == 2 && read_count[0] >= 2 * stim.iter ? stim.wr_bank : stim.op1_bank;
      expect_addr = read_count[0] % stim.iter;
      expect_group = stim.rs2[1:0] == 2 && read_count[0] >= 2 * stim.iter ? stim.wr_group : stim.op1_group;
    end else begin
      expect_bank = stim.op2_bank;
      if (stim.rs2[1:0] == 0) begin
        row = read_count[1] / 16;
        lane = read_count[1] % 16;
        input_byte = stim.src_words[row][lane*8+:8];
        expect_group = stim.op2_group + (stim.op2_col == 4 ? lane / 4 : 0);
        expect_addr = (stim.op2_col == 4 ? (lane % 4) * 16 : 0) + input_byte[7:4];
      end else begin
        row = read_count[1] / 4;
        lane = read_count[1] % 4;
        input_word = $signed(stim.src_words[row][lane*32+:32]);
        if (stim.rs2[1:0] == 1) begin
          if (input_word < -512) input_word = -512;
          if (input_word > 511) input_word = 511;
          index = $unsigned(input_word) & 1023;
        end else begin
          distance = longint'(maximum) - longint'(input_word);
          index = distance > 1023 ? 1023 : int'(distance);
        end
        expect_group = stim.op2_group;
        expect_addr  = index / 4;
      end
    end
    if (item.bank_id !== expect_bank[BB_BLINK_BANK_ID_W-1:0])
      `uvm_fatal("READ", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, expect_bank))
    if (item.group_id !== expect_group[BB_BLINK_GROUP_ID_W-1:0])
      `uvm_fatal("READ", $sformatf("group mismatch: got %0d exp %0d", item.group_id, expect_group))
    if (item.addr !== expect_addr[BB_BLINK_BANK_ADDR_W-1:0])
      `uvm_fatal("READ", $sformatf("addr mismatch: got %0d exp %0d", item.addr, expect_addr))
    read_count[item.port]++;
  endfunction

  function void write_write(bb_blink_write_item item);
    lut_cmd_item stim;
    int unsigned expect_addr;
    bit [127:0] expected_data;
    int signed input_word;
    int unsigned index;
    longint signed distance;
    stim = current_stim("WRITE");
    if (item.port != 0 || item.bank_id !== stim.wr_bank ||
        item.rob_id !== stim.rob_id || item.group_id !== stim.wr_group || item.mask !== 16'hFFFF)
      `uvm_fatal("WRITE", "write route or mask mismatch")
    if (write_count >= expected_writes)
      `uvm_fatal("WRITE", $sformatf("extra write %0d", write_count))
    expect_addr = stim.rs2[1:0] == 2 && write_count >= stim.iter ? write_count - stim.iter : write_count;
    if (item.addr !== expect_addr[BB_BLINK_BANK_ADDR_W-1:0])
      `uvm_fatal("WRITE", $sformatf("addr mismatch: got %0d exp %0d", item.addr, expect_addr))
    if (stim.rs2[1:0] == 2 && write_count < stim.iter) begin
      for (int lane = 0; lane < 4; lane++) begin
        input_word = $signed(stim.src_words[write_count][lane*32+:32]);
        distance = longint'(maximum) - longint'(input_word);
        index = distance > 1023 ? 1023 : int'(distance);
        expected_data[lane*32+:32] = stim.lut_words[index/4][(index%4)*32+:32];
      end
    end else begin
      expected_data = stim.dst_words[expect_addr];
    end
    if (item.data !== expected_data)
      `uvm_fatal(
          "SCB", $sformatf(
          "data mismatch at addr %0d: got 0x%032h exp 0x%032h", item.addr, item.data, expected_data
          ))
    write_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    lut_cmd_item stim;

    stim = current_stim("RESP");
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    resp_count++;
  endfunction

  function lut_cmd_item current_stim(string tag);
    if (stim_q.size() == 0) begin
      `uvm_fatal("SCB", $sformatf("%s observed before stimulus", tag))
    end
    return stim_q[0];
  endfunction

  function bit done();
    if (cmd_count != 1 || write_count != expected_writes || resp_count != 1) return 1'b0;
    for (int p = 0; p < `BB_IN_BW; p++) begin
      if (read_count[p] != expected_reads[p]) return 1'b0;
    end
    return 1'b1;
  endfunction

  function void reset_counters();
    stim_q.delete();
    cmd_count       = 0;
    write_count     = 0;
    resp_count      = 0;
    expected_writes = 0;
    for (int p = 0; p < `BB_IN_BW; p++) begin
      read_count[p]     = 0;
      expected_reads[p] = 0;
    end
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (!done()) begin
      `uvm_fatal("SCB", $sformatf(
                            "incomplete: cmds=%0d reads=%0d/%0d,%0d/%0d writes=%0d/%0d resp=%0d",
                            cmd_count, read_count[0], expected_reads[0], read_count[1],
                            expected_reads[1], write_count, expected_writes, resp_count))
    end
    stim_q.delete();
  endfunction
endclass
