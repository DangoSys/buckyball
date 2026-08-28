class int2fp_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(int2fp_scoreboard)

  uvm_analysis_imp_stim #(bb_blink_cmd_item, int2fp_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(bb_blink_cmd_item, int2fp_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(bb_blink_read_item, int2fp_scoreboard) read_imp;
  uvm_analysis_imp_write #(bb_blink_write_item, int2fp_scoreboard) write_imp;
  uvm_analysis_imp_resp #(bb_blink_resp_item, int2fp_scoreboard) resp_imp;
  uvm_analysis_imp_mmio_read #(bb_mmio_read_item, int2fp_scoreboard) mmio_read_imp;

  bb_blink_mem_model #(1, 1) mem_model;

  int2fp_cmd_item stim_q[$];
  bit [127:0] expected_words[INT2FP_MAX_WORDS];
  int unsigned expected_reads;
  int unsigned expected_writes;
  int unsigned cmd_count;
  int unsigned read_count;
  int unsigned write_count;
  int unsigned resp_count;
  int unsigned mmio_read_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_imp = new("stim_imp", this);
    cmd_imp = new("cmd_imp", this);
    read_imp = new("read_imp", this);
    write_imp = new("write_imp", this);
    resp_imp = new("resp_imp", this);
    mmio_read_imp = new("mmio_read_imp", this);
  endfunction

  function void write_stim(bb_blink_cmd_item item);
    int2fp_cmd_item fitem;
    int2fp_cmd_item clone;
    int i;
    int group;

    if (stim_q.size() != 0) begin
      `uvm_fatal("SCB", "single outstanding command supported")
    end
    if (!$cast(fitem, item)) begin
      `uvm_fatal("SCB", "stim item is not int2fp_cmd_item")
    end
    if (!$cast(clone, fitem.clone())) begin
      `uvm_fatal("SCB", "failed to clone stimulus item")
    end
    stim_q.push_back(clone);

    if (mem_model == null) begin
      `uvm_fatal("SCB", "mem_model handle not set")
    end
    mem_model.clear_mem();
    mem_model.load_mmio_word(int'(clone.da_addr), clone.da_bits);
    if (clone.per_channel) begin
      for (i = 0; i < clone.op1_col * 4; i++) begin
        mem_model.load_mmio_word(int'(clone.dw_addr) + i * 4, clone.dw_bits[i]);
      end
    end else begin
      mem_model.load_mmio_word(int'(clone.dw_addr), clone.dw_bits[0]);
    end
    for (i = 0; i < clone.iter; i++) begin
      for (group = 0; group < clone.op1_col; group++) begin
        mem_model.load_word_g(int'(clone.op1_bank), group, i,
                              clone.input_words[i*clone.op1_col+group]);
      end
    end
    mem_model.arm();

    build_expected(clone);
    expected_writes = clone.iter * clone.op1_col;
    expected_reads  = clone.iter * clone.op1_col;
  endfunction

  function void build_expected(int2fp_cmd_item item);
    for (int w = 0; w < item.iter; w++) begin
      for (int group = 0; group < item.op1_col; group++) begin
        for (int e = 0; e < 4; e++) begin
          int signed v = $signed(item.input_words[w*item.op1_col+group][e*32+:32]);
          int scale_index = item.per_channel ? group * 4 + e : 0;
          expected_words[w*item.op1_col+group][e*32+:32] =
              int2fp_ref_fp32(v, item.da_bits, item.dw_bits[scale_index]);
        end
      end
    end
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    int2fp_cmd_item stim;
    stim = current_stim("CMD");
    check_cmd(item, stim);
    cmd_count++;
  endfunction

  function void check_cmd(bb_blink_cmd_item got, int2fp_cmd_item exp);
    if (got.bid !== exp.bid)
      `uvm_fatal("CMD", $sformatf("bid mismatch: got %0d exp %0d", got.bid, exp.bid))
    if (got.funct7 !== exp.funct7)
      `uvm_fatal("CMD", $sformatf("funct7 mismatch: got %0d exp %0d", got.funct7, exp.funct7))
    if (got.iter !== exp.iter)
      `uvm_fatal("CMD", $sformatf("iter mismatch: got %0d exp %0d", got.iter, exp.iter))
    if (got.special[12:0] !== exp.da_addr ||
        got.special[25:13] !== exp.dw_addr || got.special[63:26] !== '0)
      `uvm_fatal("CMD", "Int2Fp special must contain only MMIO scale metadata")
    if (got.funct7 !== (exp.per_channel ? INT2FP_CHANNEL_CORE_FUNCT7[6:0]
                                        : INT2FP_TENSOR_CORE_FUNCT7[6:0]))
      `uvm_fatal("CMD", "Int2Fp funct7 does not match scale mode")
    if (got.op1_bank !== exp.op1_bank || got.wr_bank !== exp.wr_bank)
      `uvm_fatal("CMD", "bank field mismatch")
    if (got.op1_col !== exp.op1_col || got.wr_col !== exp.wr_col)
      `uvm_fatal("CMD", "column field mismatch")
    if (got.rob_id !== exp.rob_id)
      `uvm_fatal("CMD", $sformatf("rob_id mismatch: got %0d exp %0d", got.rob_id, exp.rob_id))
  endfunction

  function void write_read(bb_blink_read_item item);
    int2fp_cmd_item stim;
    int unsigned expect_addr;

    stim = current_stim("READ");
    if (item.bank_id !== stim.op1_bank)
      `uvm_fatal("READ", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.op1_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("READ", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))

    if (item.group_id !== (read_count % stim.op1_col)) `uvm_fatal("READ", "group mismatch")
    if (item.addr !== (read_count / stim.op1_col)) `uvm_fatal("READ", "addr mismatch")

    read_count++;
  endfunction

  function void write_write(bb_blink_write_item item);
    int2fp_cmd_item stim;

    stim = current_stim("WRITE");
    if (item.bank_id !== stim.wr_bank)
      `uvm_fatal("WRITE", $sformatf("bank mismatch: got %0d exp %0d", item.bank_id, stim.wr_bank))
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("WRITE", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.group_id !== (write_count % stim.op1_col)) `uvm_fatal("WRITE", "group mismatch")
    if (item.addr !== (write_count / stim.op1_col)) `uvm_fatal("WRITE", "addr mismatch")
    if (item.mask !== 16'hFFFF)
      `uvm_fatal("WRITE", $sformatf("mask mismatch: got 0x%04h", item.mask))
    if (item.data !== expected_words[item.addr*stim.op1_col+item.group_id])
      `uvm_fatal("SCB", $sformatf(
                 "data mismatch at addr %0d: got 0x%032h exp 0x%032h",
                 item.addr,
                 item.data,
                 expected_words[item.addr*stim.op1_col+item.group_id]
                 ))
    write_count++;
  endfunction

  function void write_mmio_read(bb_mmio_read_item item);
    int2fp_cmd_item stim;
    int unsigned expected_addr;
    stim = current_stim("MMIO_READ");
    if (mmio_read_count < 4) expected_addr = stim.da_addr + mmio_read_count;
    else if (!stim.per_channel) expected_addr = stim.dw_addr + (mmio_read_count - 4);
    else expected_addr = stim.dw_addr + ((mmio_read_count - 4) % (stim.op1_col * 16));
    if (item.addr !== expected_addr) `uvm_fatal("MMIO", "scale read address mismatch")
    mmio_read_count++;
  endfunction

  function void write_resp(bb_blink_resp_item item);
    int2fp_cmd_item stim;

    stim = current_stim("RESP");
    if (item.rob_id !== stim.rob_id)
      `uvm_fatal("RESP", $sformatf("rob_id mismatch: got %0d exp %0d", item.rob_id, stim.rob_id))
    if (item.is_sub !== 1'b0) `uvm_fatal("RESP", "is_sub should be 0")
    if (item.sub_rob_id !== 8'h00)
      `uvm_fatal("RESP", $sformatf("sub_rob_id mismatch: got 0x%0h", item.sub_rob_id))
    resp_count++;
  endfunction

  function int2fp_cmd_item current_stim(string tag);
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
           resp_count == 1 && mmio_read_count ==
             (stim_q[0].per_channel ? 4 + expected_reads * 16 : 8);
  endfunction

  function void reset_counters();
    stim_q.delete();
    cmd_count = 0;
    read_count = 0;
    write_count = 0;
    resp_count = 0;
    mmio_read_count = 0;
    expected_reads = 0;
    expected_writes = 0;
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
