class im2col_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(im2col_cmd_item)

  // Input rows to preload, the expected reads, and the expected writes, each
  // as (group, line[, data]) in the order the Ball issues them.
  int unsigned src_group[$];
  int unsigned src_addr[$];
  bit [127:0] src_data[$];
  int unsigned read_group[$];
  int unsigned read_addr[$];
  int unsigned dst_group[$];
  int unsigned dst_addr[$];
  bit [127:0] dst_data[$];

  function new(string name = "im2col_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_case(int unsigned index);
    im2col_cmd_dpi_t cmd;

    if (im2col_case_load(index) != 0) `uvm_fatal("CASE", $sformatf("im2col case %0d failed", index))
    im2col_case_cmd(cmd);

    // bb_im2col: banks and input size come from rs1, the window geometry from rs2.
    rob_id       = cmd.rob_id[BB_BLINK_ROB_ID_W-1:0];
    funct        = IM2COL_THREAD_FUNCT7[6:0];
    rs2          = {cmd.rs2_hi, cmd.rs2_lo};
    iter         = cmd.iter;
    op1_bank     = cmd.op1_bank;
    op2_bank     = '0;
    wr_bank      = cmd.wr_bank;
    op1_group    = cmd.op1_group;
    op2_group    = '0;
    wr_group     = cmd.wr_group;
    op1_col      = cmd.op1_col;
    op2_col      = '0;
    wr_col       = cmd.wr_col;
    wr_read_bank = '0;
    wr_arch_bank = cmd.wr_bank;

    for (int i = 0; i < cmd.num_src; i++) begin
      src_group.push_back(im2col_case_src_group(i));
      src_addr.push_back(im2col_case_src_addr(i));
      src_data.push_back({im2col_case_src_hi(i), im2col_case_src_lo(i)});
    end
    for (int i = 0; i < cmd.num_reads; i++) begin
      read_group.push_back(im2col_case_read_group(i));
      read_addr.push_back(im2col_case_read_addr(i));
    end
    for (int i = 0; i < cmd.num_dst; i++) begin
      dst_group.push_back(im2col_case_dst_group(i));
      dst_addr.push_back(im2col_case_dst_addr(i));
      dst_data.push_back({im2col_case_dst_hi(i), im2col_case_dst_lo(i)});
    end
  endfunction

  function void do_copy(uvm_object rhs);
    im2col_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) `uvm_fatal("COPY", "rhs is not im2col_cmd_item")
    src_group  = rhs_.src_group;
    src_addr   = rhs_.src_addr;
    src_data   = rhs_.src_data;
    read_group = rhs_.read_group;
    read_addr  = rhs_.read_addr;
    dst_group  = rhs_.dst_group;
    dst_addr   = rhs_.dst_addr;
    dst_data   = rhs_.dst_data;
  endfunction
endclass
