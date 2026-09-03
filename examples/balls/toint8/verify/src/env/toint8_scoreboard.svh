class toint8_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(toint8_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, toint8_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, toint8_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, toint8_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, toint8_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, toint8_scoreboard) resp_imp;
  uvm_analysis_imp_mmio_write #(bb_mmio_write_item, toint8_scoreboard) mmio_write_imp;

  bb_blink_mem_model #(1, 1) mem_model;

  toint8_cmd_item stim_q[$];
  bit [127:0] expected_words[FP2INT_MAX_OUTPUT_WORDS];
  int unsigned expected_reads;
  int unsigned expected_writes;
  int unsigned cmd_count;
  int unsigned read_count;
  int unsigned write_count;
  int unsigned resp_count;
  int unsigned mmio_write_count;
  int unsigned expect_group;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_imp = new("stim_imp", this);
    cmd_imp = new("cmd_imp", this);
    read_imp = new("read_imp", this);
    write_imp = new("write_imp", this);
    resp_imp = new("resp_imp", this);
    mmio_write_imp = new("mmio_write_imp", this);
  endfunction

  function void write_stim(bb_blink_cmd_item item);
    toint8_cmd_item fitem;
    toint8_cmd_item clone;
    int row;
    int group;

    if (stim_q.size() != 0) begin
      `uvm_fatal("SCB", "single outstanding command supported")
    end
    if (!$cast(fitem, item)) begin
      `uvm_fatal("SCB", "stim item is not toint8_cmd_item")
    end
    if (!$cast(clone, fitem.clone())) begin
      `uvm_fatal("SCB", "failed to clone stimulus item")
    end
    stim_q.push_back(clone);

    if (mem_model == null) begin
      `uvm_fatal("SCB", "mem_model handle not set")
    end
    mem_model.clear_mem();
    for (row = 0; row < clone.iter; row++) begin
      for (group = 0; group < clone.op1_col; group++) begin
        mem_model.load_word_g(int'(clone.op1_bank), group, row,
                              clone.input_words[row*clone.op1_col+group]);
      end
    end
    mem_model.arm();

    expected_writes = clone.num_src_words / FP2INT_NUM_WORDS;
    expected_reads  = clone.num_src_words * 2;
    build_expected(clone);
    expect_group = 0;
  endfunction

  function void build_expected(toint8_cmd_item item);
    bit [31:0] quant_scale_bits;

    quant_scale_bits = toint8_quant_scale_bits(item.da_bits);
    for (int word = 0; word < expected_writes; word++) begin
      bit [127:0] packed_word = '0;
      for (int source_word = 0; source_word < FP2INT_NUM_WORDS; source_word++) begin
        bit [127:0] src = item.input_words[word*FP2INT_NUM_WORDS+source_word];
        for (int lane = 0; lane < 4; lane++) begin
          bit [31:0] fp_bits = src[lane*32+:32];
          bit [ 7:0] q = toint8_ref_i8(fp_bits, quant_scale_bits);
          packed_word[source_word*32+lane*8+:8] = q;
        end
      end
      expected_words[word] = packed_word;
    end
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    toint8_cmd_item stim;
    stim = current_stim("CMD");
    check_cmd(item, stim);
    cmd_count++;
  endfunction

  function void check_cmd(bb_blink_cmd_item got, toint8_cmd_item exp);
    if (got.bid !== exp.bid)
      `uvm_fatal("CMD", $sformatf("bid mismatch: got %0d exp %0d", got.bid, exp.bid))
    if (got.funct7 !== exp.funct7)
      `uvm_fatal("CMD", $sformatf("funct7 mismatch: got %0d exp %0d", got.funct7, exp.funct7))
    if (got.iter !== exp.iter)
      `uvm_fatal("CMD", $sformatf("iter mismatch: got %0d exp %0d", got.iter, exp.iter))
    if (got.special[12:0] !== exp.da_addr || got.special[63:13] !== '0)
      `uvm_fatal("CMD", "ToInt8 special must contain only Da MMIO address")
    if (got.op1_bank !== exp.op1_bank || got.wr_bank !== exp.wr_bank)
      `uvm_fatal("CMD", "bank field mismatch")
    if (got.op1_col !== exp.op1_col || got.wr_col !== exp.wr_col)
      `uvm_fatal("CMD", "column field mismatch")
    if (got.rob_id !== exp.rob_id)
      `uvm_fatal("CMD", $sformatf("rob_id mismatch: got %0d exp %0d", got.rob_id, exp.rob_id))
  endfunction

  function void write_read(bb_blink_read_item item);
    toint8_cmd_item stim;
    int unsigned expect_addr;

    stim = current_stim("READ");
    if (item.bank_id !== stim.op1_bank)
      `uvm_fatal("READ", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.op1_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))

    expect_addr  = (read_count % stim.num_src_words) / stim.op1_col;
    expect_group = (read_count % stim.num_src_words) % stim.op1_col;
    if (item.group_id !== expect_group[4:0])
      `uvm_fatal("READ", $sformatf("group mismatch: got %0d exp %0d", item.group_id, expect_group))
    if (item.addr !== expect_addr[6:0])
      `uvm_fatal("READ", $sformatf("addr mismatch: got %0d exp %0d", item.addr, expect_addr))

    read_count++;
  endfunction

  function void write_write(bb_blink_write_item item);
    toint8_cmd_item stim;

    stim = current_stim("WRITE");
    if (item.bank_id !== stim.wr_bank)
      `uvm_fatal("WRITE", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.wr_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("WRITE", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.group_id !== (write_count % stim.wr_col))
      `uvm_fatal("WRITE", $sformatf(
                 "group mismatch: got %0d exp %0d", item.group_id, write_count % stim.wr_col))
    if (item.addr !== (write_count / stim.wr_col))
      `uvm_fatal("WRITE", $sformatf(
                 "addr mismatch: got %0d exp %0d", item.addr, write_count / stim.wr_col))
    if (item.mask !== 16'hFFFF)
      `uvm_fatal("WRITE", $sformatf("mask mismatch: got 0x%04h", item.mask))
    if (item.data !== expected_words[write_count])
      `uvm_fatal("SCB", $sformatf(
                 "data mismatch at addr %0d: got 0x%032h exp 0x%032h",
                 write_count,
                 item.data,
                 expected_words[write_count]
                 ))
    write_count++;
  endfunction

  function void write_mmio_write(bb_mmio_write_item item);
    toint8_cmd_item stim;
    stim = current_stim("MMIO_WRITE");
    if (item.addr !== stim.da_addr + mmio_write_count)
      `uvm_fatal("MMIO", "Da write address mismatch")
    if (item.data !== stim.da_bits[mmio_write_count*8+:8])
      `uvm_fatal("MMIO", "Da write data mismatch")
    mmio_write_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    toint8_cmd_item stim;

    stim = current_stim("RESP");
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.is_sub !== 1'b0) `uvm_fatal("RESP", "is_sub should be 0")
    if (item.sub_rob_id !== 8'h00)
      `uvm_fatal("RESP", $sformatf("sub_rob_id mismatch: got 0x%0h", item.sub_rob_id))
    resp_count++;
  endfunction

  function toint8_cmd_item current_stim(string tag);
    if (stim_q.size() == 0) begin
      `uvm_fatal("SCB", $sformatf("%s observed before stimulus", tag))
      return null;
    end
    return stim_q[0];
  endfunction

  function bit done();
    return cmd_count == 1 &&
           read_count == expected_reads &&
           write_count == expected_writes &&
           resp_count == 1 && mmio_write_count == 4;
  endfunction

  function void reset_counters();
    stim_q.delete();
    cmd_count = 0;
    read_count = 0;
    write_count = 0;
    resp_count = 0;
    mmio_write_count = 0;
    expected_reads = 0;
    expected_writes = 0;
    expect_group = 0;
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (!done()) begin
      `uvm_fatal("SCB", $sformatf("incomplete: cmds=%0d reads=%0d/%0d writes=%0d/%0d resp=%0d",
                                  cmd_count, read_count, expected_reads, write_count,
                                  expected_writes, resp_count))
    end
    stim_q.delete();
  endfunction
endclass
