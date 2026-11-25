class mxint8quant_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(mxint8quant_cmd_item)

  int unsigned preload_addr[];
  bit [127:0] preload_data[];
  int unsigned write_addr[];
  int unsigned write_kind[];
  bit [127:0] write_data[];
  int unsigned events[];
  int unsigned hold;

  function new(string name = "mxint8quant_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_step(int unsigned step);
    mxint8quant_cmd_dpi_t cmd;
    int unsigned n;

    mxint8quant_case_cmd(step, cmd);
    rob_id       = cmd.rob_id[BB_BLINK_ROB_ID_W-1:0];
    funct        = MXINT8QUANT_FUNCT7_VALUE[6:0];
    rs2          = {cmd.rs2_hi, cmd.rs2_lo};
    iter         = cmd.iter;
    op1_bank     = cmd.op1_bank;
    op2_bank     = cmd.op2_bank;
    wr_bank      = cmd.wr_bank;
    op1_group    = cmd.op1_group;
    op2_group    = cmd.op2_group;
    wr_group     = cmd.wr_group;
    op1_col      = cmd.op1_col;
    op2_col      = cmd.op2_col;
    wr_col       = cmd.wr_col;
    wr_read_bank = '0;
    wr_arch_bank = cmd.wr_bank;
    if (rob_id != cmd.rob_id || op1_bank != cmd.op1_bank || wr_bank != cmd.wr_bank ||
        op1_group != cmd.op1_group || wr_group != cmd.wr_group)
      `uvm_fatal("CASE", "command field exceeds the configured BLink width")
    hold = mxint8quant_case_hold(step);

    n = mxint8quant_case_preload_count(step);
    preload_addr = new[n];
    preload_data = new[n];
    for (int unsigned i = 0; i < n; i++) begin
      preload_addr[i] = mxint8quant_case_preload_addr(step, i);
      preload_data[i] = {
        mxint8quant_case_preload_hi(step, i), mxint8quant_case_preload_lo(step, i)
      };
    end
    n = mxint8quant_case_write_count(step);
    write_addr = new[n];
    write_kind = new[n];
    write_data = new[n];
    for (int unsigned i = 0; i < n; i++) begin
      write_addr[i] = mxint8quant_case_write_addr(step, i);
      write_kind[i] = mxint8quant_case_write_kind(step, i);
      write_data[i] = {mxint8quant_case_write_hi(step, i), mxint8quant_case_write_lo(step, i)};
    end
    events = new[mxint8quant_num_events()];
    foreach (events[i]) events[i] = mxint8quant_case_event(step, i);
  endfunction

  function void do_copy(uvm_object rhs);
    mxint8quant_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) `uvm_fatal("COPY", "rhs is not mxint8quant_cmd_item")
    preload_addr = rhs_.preload_addr;
    preload_data = rhs_.preload_data;
    write_addr   = rhs_.write_addr;
    write_kind   = rhs_.write_kind;
    write_data   = rhs_.write_data;
    events       = rhs_.events;
    hold         = rhs_.hold;
  endfunction
endclass
