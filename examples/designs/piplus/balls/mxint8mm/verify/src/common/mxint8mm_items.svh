class mxint8mm_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(mxint8mm_cmd_item)

  int unsigned step;
  bit read_old;
  bit update;
  bit stall;
  int unsigned features;
  int unsigned preload_bank[];
  int unsigned preload_group[];
  int unsigned preload_addr[];
  bit [127:0] preload_data[];
  int unsigned write_addr[];
  bit [127:0] write_data[];

  function new(string name = "mxint8mm_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_step(int unsigned step_index);
    mxint8mm_cmd_dpi_t cmd;
    int unsigned n;
    longint unsigned lo, hi;

    mxint8mm_case_cmd(step_index, cmd);
    step         = step_index;
    rob_id       = cmd.rob_id[BB_BLINK_ROB_ID_W-1:0];
    funct        = `MXINT8MM_FUNCT7;
    rs2          = {cmd.rs2_hi, cmd.rs2_lo};
    iter         = cmd.iter;
    op1_bank     = cmd.a_bank;
    op2_bank     = cmd.b_bank;
    wr_bank      = cmd.c_bank;
    op1_group    = cmd.a_group;
    op2_group    = cmd.b_group;
    wr_group     = cmd.c_group;
    op1_col      = 1;
    op2_col      = 1;
    wr_col       = 1;
    wr_read_bank = cmd.read_bank;
    wr_arch_bank = cmd.c_bank;
    read_old     = rs2[62] || rs2[63];
    update       = cmd.update;
    stall        = cmd.stall;
    features     = cmd.features;
    if (rob_id != cmd.rob_id || op1_bank != cmd.a_bank || op2_bank != cmd.b_bank ||
        wr_bank != cmd.c_bank || wr_read_bank != cmd.read_bank || op1_group != cmd.a_group ||
        op2_group != cmd.b_group || wr_group != cmd.c_group)
      `uvm_fatal("CASE", "command field exceeds the configured BLink width")

    n = mxint8mm_case_preload_count(step_index);
    preload_bank = new[n];
    preload_group = new[n];
    preload_addr = new[n];
    preload_data = new[n];
    for (int unsigned i = 0; i < n; i++) begin
      mxint8mm_case_preload(step_index, i, preload_bank[i], preload_group[i], preload_addr[i], lo,
                            hi);
      preload_data[i] = {hi, lo};
    end
    n = mxint8mm_case_write_count(step_index);
    write_addr = new[n];
    write_data = new[n];
    for (int unsigned i = 0; i < n; i++) begin
      mxint8mm_case_write(step_index, i, write_addr[i], lo, hi);
      write_data[i] = {hi, lo};
    end
  endfunction

  function void do_copy(uvm_object rhs);
    mxint8mm_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) `uvm_fatal("COPY", "rhs is not mxint8mm_cmd_item")
    step          = rhs_.step;
    read_old      = rhs_.read_old;
    update        = rhs_.update;
    stall         = rhs_.stall;
    features      = rhs_.features;
    preload_bank  = rhs_.preload_bank;
    preload_group = rhs_.preload_group;
    preload_addr  = rhs_.preload_addr;
    preload_data  = rhs_.preload_data;
    write_addr    = rhs_.write_addr;
    write_data    = rhs_.write_data;
  endfunction
endclass
