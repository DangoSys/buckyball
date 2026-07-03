class fp2int_scoreboard extends uvm_scoreboard;
  `uvm_component_utils(fp2int_scoreboard)

  uvm_analysis_imp_stim #(fp2int_cmd_item, fp2int_scoreboard) stim_imp;
  uvm_analysis_imp_cmd #(fp2int_cmd_item, fp2int_scoreboard) cmd_imp;
  uvm_analysis_imp_read #(fp2int_read_item, fp2int_scoreboard) read_imp;
  uvm_analysis_imp_write #(fp2int_write_item, fp2int_scoreboard) write_imp;
  uvm_analysis_imp_resp #(fp2int_resp_item, fp2int_scoreboard) resp_imp;

  fp2int_cmd_item stim_q[$];
  bit [127:0] expected_words[FP2INT_NUM_WORDS];
  int unsigned cmd_count;
  int unsigned read_count;
  int unsigned write_count;
  int unsigned resp_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    stim_imp  = new("stim_imp", this);
    cmd_imp   = new("cmd_imp", this);
    read_imp  = new("read_imp", this);
    write_imp = new("write_imp", this);
    resp_imp  = new("resp_imp", this);
  endfunction

  function void write_stim(fp2int_cmd_item item);
    fp2int_cmd_item clone;

    if (stim_q.size() != 0) begin
      `uvm_fatal("SCB", "this directed scoreboard supports one outstanding command")
    end
    if (!$cast(clone, item.clone())) begin
      `uvm_fatal("SCB", "failed to clone stimulus item")
    end
    if (clone.iter != FP2INT_NUM_WORDS) begin
      `uvm_fatal("SCB", $sformatf("unsupported iter in this test: %0d", clone.iter))
    end

    build_expected(clone);
    stim_q.push_back(clone);
  endfunction

  function void build_expected(fp2int_cmd_item item);
    for (int w = 0; w < FP2INT_NUM_WORDS; w++) begin
      for (int e = 0; e < 4; e++) begin
        expected_words[w][e*32+:32] =
            fp2int_ref_i32(item.input_words[w][e*32+:32], item.scale_bits);
      end
    end
  endfunction

  function void write_cmd(fp2int_cmd_item item);
    fp2int_cmd_item stim;

    stim = current_stim("CMD");
    check_cmd(item, stim);
    cmd_count++;
  endfunction

  function void check_cmd(fp2int_cmd_item got, fp2int_cmd_item exp);
    if (got.bid !== exp.bid) begin
      `uvm_fatal("CMD", $sformatf("bid mismatch: got %0d expected %0d", got.bid, exp.bid))
    end
    if (got.funct7 !== exp.funct7) begin
      `uvm_fatal("CMD", $sformatf("funct7 mismatch: got %0d expected %0d", got.funct7, exp.funct7))
    end
    if (got.iter !== exp.iter) begin
      `uvm_fatal("CMD", $sformatf("iter mismatch: got %0d expected %0d", got.iter, exp.iter))
    end
    if (got.scale_bits !== exp.scale_bits) begin
      `uvm_fatal("CMD", $sformatf("scale mismatch: got 0x%08h expected 0x%08h", got.scale_bits,
                                  exp.scale_bits))
    end
    if (got.op1_bank !== exp.op1_bank || got.wr_bank !== exp.wr_bank) begin
      `uvm_fatal("CMD", "bank field mismatch")
    end
    if (got.op1_col !== exp.op1_col || got.wr_col !== exp.wr_col) begin
      `uvm_fatal("CMD", "column field mismatch")
    end
    if (got.rob_id !== exp.rob_id || got.is_sub !== exp.is_sub || got.sub_rob_id !== exp.sub_rob_id) begin
      `uvm_fatal("CMD", "rob field mismatch")
    end
  endfunction

  function void write_read(fp2int_read_item item);
    fp2int_cmd_item stim;

    stim = current_stim("READ");
    if (item.bank_id !== stim.op1_bank) begin
      `uvm_fatal("READ", $sformatf("read bank mismatch: got %0d expected %0d", item.bank_id,
                                   stim.op1_bank))
    end
    if (item.group_id !== 5'd0) begin
      `uvm_fatal("READ", $sformatf("read group mismatch: got %0d", item.group_id))
    end
    if (item.rob_id !== stim.rob_id) begin
      `uvm_fatal("READ", $sformatf("read rob_id mismatch: got 0x%0h expected 0x%0h", item.rob_id,
                                   stim.rob_id))
    end
    if (item.addr >= FP2INT_NUM_WORDS) begin
      `uvm_fatal("READ", $sformatf("read addr out of range: %0d", item.addr))
    end
    if (item.addr !== read_count[6:0]) begin
      `uvm_fatal("READ", $sformatf("read addr mismatch: got %0d expected %0d", item.addr,
                                   read_count))
    end

    read_count++;
  endfunction

  function void write_write(fp2int_write_item item);
    fp2int_cmd_item stim;

    stim = current_stim("WRITE");
    if (item.bank_id !== stim.wr_bank) begin
      `uvm_fatal("WRITE", $sformatf("write bank mismatch: got %0d expected %0d", item.bank_id,
                                    stim.wr_bank))
    end
    if (item.rob_id !== stim.rob_id) begin
      `uvm_fatal("WRITE", $sformatf("write rob_id mismatch: got 0x%0h expected 0x%0h", item.rob_id,
                                    stim.rob_id))
    end
    if (item.addr >= FP2INT_NUM_WORDS) begin
      `uvm_fatal("WRITE", $sformatf("write addr out of range: %0d", item.addr))
    end
    if (item.addr !== write_count[6:0]) begin
      `uvm_fatal("WRITE", $sformatf("write addr mismatch: got %0d expected %0d", item.addr,
                                    write_count))
    end
    if (item.mask !== 16'hFFFF) begin
      `uvm_fatal("WRITE", $sformatf("write mask mismatch: got 0x%04h", item.mask))
    end
    if (item.data !== expected_words[item.addr]) begin
      `uvm_fatal("SCB", $sformatf("write data mismatch at addr %0d: got 0x%032h expected 0x%032h",
                                  item.addr, item.data, expected_words[item.addr]))
    end

    write_count++;
  endfunction

  function void write_resp(fp2int_resp_item item);
    fp2int_cmd_item stim;

    stim = current_stim("RESP");
    if (item.rob_id !== stim.rob_id) begin
      `uvm_fatal("RESP", $sformatf("resp rob_id mismatch: got 0x%0h expected 0x%0h", item.rob_id,
                                   stim.rob_id))
    end
    if (item.is_sub !== stim.is_sub) begin
      `uvm_fatal("RESP", "resp is_sub mismatch")
    end
    if (item.sub_rob_id !== stim.sub_rob_id) begin
      `uvm_fatal("RESP", $sformatf("resp sub_rob_id mismatch: got 0x%0h", item.sub_rob_id))
    end

    resp_count++;
  endfunction

  function fp2int_cmd_item current_stim(string tag);
    if (stim_q.size() == 0) begin
      `uvm_fatal("SCB", $sformatf("%s observed before stimulus", tag))
      return null;
    end
    return stim_q[0];
  endfunction

  function bit done();
    return cmd_count == 1 &&
           read_count == FP2INT_NUM_WORDS &&
           write_count == FP2INT_NUM_WORDS &&
           resp_count == 1;
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (!done()) begin
      `uvm_fatal("SCB", $sformatf("incomplete test: cmds=%0d reads=%0d writes=%0d responses=%0d",
                                  cmd_count, read_count, write_count, resp_count))
    end
  endfunction
endclass
