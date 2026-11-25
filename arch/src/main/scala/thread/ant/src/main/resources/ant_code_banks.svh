// Reuse the same 16 KiB TSS for two code segments, then execute across their boundary.
task automatic code_banks();
  for (int c = 0; c < 2; c++) begin
    logic [127:0] word, ignored;
    logic [127:0] low_tail = {32'h00000013, 32'h00000013, 32'h00000013, 32'h03100513};
    logic [127:0] high_head = {32'h00000013, 32'h00000013, 32'h00000073, 32'h00000893};
    int origin = 16384 - 16;
    for (int bank = 0; bank < 2; bank++) begin
      for (int offset = 0; offset < 16384; offset += 16) begin
        word = bank ? {4{32'h00100013}} : {4{32'h00000013}};
        if (!bank && offset == 16368) word = low_tail;
        if (bank && offset == 0) word = high_head;
        host_access(`TSS_BASE + offset, 1, word, ignored);
      end
      move(c, 1, 1, bank * 16384, 0, 16384);
    end
    // Reading back both complete leaves also detects premature reuse of the TSS.
    for (int bank = 0; bank < 2; bank++) begin
      move(c, 1, 0, bank * 16384, 0, 16384);
      for (int offset = 0; offset < 16384; offset += 16) begin
        logic [127:0] expected_word = bank ? {4{32'h00100013}} : {4{32'h00000013}};
        if (!bank && offset == 16368) expected_word = low_tail;
        if (bank && offset == 0) expected_word = high_head;
        host_access(`TSS_BASE + offset, 0, 0, word);
        evidence::check(word === expected_word, "segmented code load/readback");
      end
    end
    task_id[c]++;
    entry[c] = origin;
    code_end[c] = 16384 + 8;
    argument[c] = `TLS_BASE;
    stack[c] = `TLS_END;
    in_use = 1;
    launch(c);
    finish(c, 49);
    // Cancel a fetch from the upper leaf, drain it, and refill that same leaf.
    put(c * 2, 16384, {96'b0, 32'h0000006f}, 1);
    task_id[c]++;
    entry[c] = 16384;
    code_end[c] = 16388;
    launch(c);
    do
    @(negedge clock);
    while ((c ? dut.ants_1.executor.state : dut.ants_0.executor.state) != 4'h2);
    cancel[c] = 1;
    @(negedge clock);
    cancel[c] = 0;
    finish(c, 0, 1);
    put(c * 2, 16384, {32'h00000013, 32'h00000073, 32'h00000893, 32'h04300513}, 1);
    task_id[c]++;
    code_end[c] = 16396;
    launch(c);
    finish(c, 67);
    in_use = 0;
    entry[c] = 0;
    code_end[c] = 16;
  end
  `uvm_info("CODE_BANKS",
            "two 16KiB code leaves: segmented load, boundary execution, cancel/refill", UVM_LOW)
endtask
