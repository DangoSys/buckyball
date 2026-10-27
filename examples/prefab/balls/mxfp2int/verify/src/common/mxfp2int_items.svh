class mxfp2int_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(mxfp2int_cmd_item)

  int unsigned preload_bank[];
  int unsigned preload_group[];
  int unsigned preload_addr[];
  bit [127:0] preload_data[];
  int unsigned read_addr0[];
  int unsigned read_addr1[];
  int unsigned write_addr[];
  bit [15:0] write_mask[];
  bit [127:0] write_data[];

  function new(string name = "mxfp2int_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_step(int unsigned step);
    mxfp2int_cmd_dpi_t cmd;
    int unsigned n;

    mxfp2int_case_cmd(step, cmd);
    rob_id       = cmd.rob_id[BB_BLINK_ROB_ID_W-1:0];
    funct        = MXFP2INT_THREAD_FUNCT7[6:0];
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
    if (rob_id != cmd.rob_id || op1_bank != cmd.op1_bank || op2_bank != cmd.op2_bank ||
        wr_bank != cmd.wr_bank || op1_group != cmd.op1_group || op2_group != cmd.op2_group ||
        wr_group != cmd.wr_group)
      `uvm_fatal("CASE", "command field exceeds the configured BLink width")

    n = mxfp2int_case_preload_count(step);
    preload_bank = new[n];
    preload_group = new[n];
    preload_addr = new[n];
    preload_data = new[n];
    for (int unsigned i = 0; i < n; i++) begin
      preload_bank[i]  = mxfp2int_case_preload_bank(step, i);
      preload_group[i] = mxfp2int_case_preload_group(step, i);
      preload_addr[i]  = mxfp2int_case_preload_addr(step, i);
      preload_data[i]  = {mxfp2int_case_preload_hi(step, i), mxfp2int_case_preload_lo(step, i)};
    end
    n = mxfp2int_case_read_count(step, 0);
    read_addr0 = new[n];
    for (int unsigned i = 0; i < n; i++) read_addr0[i] = mxfp2int_case_read_addr(step, 0, i);
    n = mxfp2int_case_read_count(step, 1);
    read_addr1 = new[n];
    for (int unsigned i = 0; i < n; i++) read_addr1[i] = mxfp2int_case_read_addr(step, 1, i);
    n = mxfp2int_case_write_count(step);
    write_addr = new[n];
    write_mask = new[n];
    write_data = new[n];
    for (int unsigned i = 0; i < n; i++) begin
      write_addr[i] = mxfp2int_case_write_addr(step, i);
      write_mask[i] = mxfp2int_case_write_mask(step, i);
      write_data[i] = {mxfp2int_case_write_hi(step, i), mxfp2int_case_write_lo(step, i)};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    mxfp2int_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) `uvm_fatal("COPY", "rhs is not mxfp2int_cmd_item")
    preload_bank  = rhs_.preload_bank;
    preload_group = rhs_.preload_group;
    preload_addr  = rhs_.preload_addr;
    preload_data  = rhs_.preload_data;
    read_addr0    = rhs_.read_addr0;
    read_addr1    = rhs_.read_addr1;
    write_addr    = rhs_.write_addr;
    write_mask    = rhs_.write_mask;
    write_data    = rhs_.write_data;
  endfunction
endclass
