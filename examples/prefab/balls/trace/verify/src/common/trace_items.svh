class trace_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(trace_cmd_item)

  function new(string name = "trace_cmd_item");
    super.new(name);
  endfunction

  // bdb_counter: rs1 is zero, so every bank field is zero; rs2 carries the
  // subcommand, counter and payload.
  function void load_rust_case(int unsigned index);
    trace_cmd_dpi_t cmd;
    trace_case_cmd(index, cmd);
    rob_id       = cmd.rob_id[BB_BLINK_ROB_ID_W-1:0];
    funct        = BDB_COUNTER_THREAD_FUNCT7[6:0];
    rs2          = {cmd.rs2_hi, cmd.rs2_lo};
    iter         = '0;
    op1_bank     = '0;
    op2_bank     = '0;
    wr_bank      = '0;
    op1_group    = '0;
    op2_group    = '0;
    wr_group     = '0;
    op1_col      = '0;
    op2_col      = '0;
    wr_col       = '0;
    wr_read_bank = '0;
    wr_arch_bank = '0;
  endfunction
endclass
