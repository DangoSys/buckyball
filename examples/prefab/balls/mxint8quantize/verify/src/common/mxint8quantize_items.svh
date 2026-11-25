class mxint8quantize_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(mxint8quantize_cmd_item)

  int unsigned preload_bank[];
  int unsigned preload_group[];
  int unsigned preload_addr[];
  bit [127:0] preload_data[];
  int unsigned read_addr[];
  int unsigned write_addr[];
  bit [15:0] write_mask[];
  bit [127:0] write_data[];
  int unsigned events[];

  function new(string name = "mxint8quantize_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_step(int unsigned step);
    mxint8quantize_cmd_dpi_t cmd;
    int unsigned n;

    mxint8quantize_case_cmd(step, cmd);
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
    if (rob_id != cmd.rob_id || op1_bank != cmd.op1_bank || op2_bank != cmd.op2_bank ||
        wr_bank != cmd.wr_bank || op1_group != cmd.op1_group || op2_group != cmd.op2_group ||
        wr_group != cmd.wr_group)
      `uvm_fatal("CASE", "command field exceeds the configured BLink width")

    n = mxint8quantize_case_preload_count(step);
    preload_bank = new[n];
    preload_group = new[n];
    preload_addr = new[n];
    preload_data = new[n];
    for (int unsigned i = 0; i < n; i++) begin
      preload_bank[i] = mxint8quantize_case_preload_bank(step, i);
      preload_group[i] = mxint8quantize_case_preload_group(step, i);
      preload_addr[i] = mxint8quantize_case_preload_addr(step, i);
      preload_data[i] = {
        mxint8quantize_case_preload_hi(step, i), mxint8quantize_case_preload_lo(step, i)
      };
    end
    n = mxint8quantize_case_read_count(step);
    read_addr = new[n];
    for (int unsigned i = 0; i < n; i++) read_addr[i] = mxint8quantize_case_read_addr(step, i);
    n = mxint8quantize_case_write_count(step);
    write_addr = new[n];
    write_mask = new[n];
    write_data = new[n];
    for (int unsigned i = 0; i < n; i++) begin
      write_addr[i] = mxint8quantize_case_write_addr(step, i);
      write_mask[i] = mxint8quantize_case_write_mask(step, i);
      write_data[i] = {
        mxint8quantize_case_write_hi(step, i), mxint8quantize_case_write_lo(step, i)
      };
    end
    n = mxint8quantize_num_events();
    events = new[n];
    for (int unsigned i = 0; i < n; i++) events[i] = mxint8quantize_case_event(step, i);
  endfunction

  function void do_copy(uvm_object rhs);
    mxint8quantize_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) `uvm_fatal("COPY", "rhs is not mxint8quantize_cmd_item")
    preload_bank  = rhs_.preload_bank;
    preload_group = rhs_.preload_group;
    preload_addr  = rhs_.preload_addr;
    preload_data  = rhs_.preload_data;
    read_addr     = rhs_.read_addr;
    write_addr    = rhs_.write_addr;
    write_mask    = rhs_.write_mask;
    write_data    = rhs_.write_data;
    events        = rhs_.events;
  endfunction
endclass
