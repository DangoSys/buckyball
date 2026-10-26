class smatmul_cmd_item extends bb_blink_cmd_item;
  `uvm_object_utils(smatmul_cmd_item)

  bit [31:0] kind;
  bit [31:0] command_index;
  bit [31:0] command_count;
  bit [31:0] op1_words;
  bit [31:0] op2_words;
  bit [31:0] k_groups;
  bit [63:0] rs1;

  function new(string name = "smatmul_cmd_item");
    super.new(name);
  endfunction

  function void load_rust_command(int unsigned index);
    matrix_cmd_dpi_t cmd;
    smatmul_case_cmd(index, cmd);
    if (cmd.kind > 1) begin
      `uvm_fatal("CASE", $sformatf("invalid command kind %0d", cmd.kind))
    end

    kind = cmd.kind;
    command_index = index;
    command_count = smatmul_case_num_commands();
    op1_words = cmd.op1_words;
    op2_words = cmd.op2_words;
    k_groups = cmd.k_groups;
    rs1 = {cmd.rs1_hi, cmd.rs1_lo};

    // BB_BANK0/1/2 and BB_ITER fields of rs1; banks are not renamed at Ball level.
    rob_id = cmd.rob_id[BB_BLINK_ROB_ID_W-1:0];
    rs2 = {cmd.rs2_hi, cmd.rs2_lo};
    iter = rs1[63:30];
    op1_bank = rs1[9:0];
    op2_bank = rs1[19:10];
    wr_bank = rs1[29:20];
    op1_group = cmd.op1_group;
    op2_group = cmd.op2_group;
    wr_group = cmd.wr_group;
    wr_read_bank = '0;
    wr_arch_bank = wr_bank;

    if (kind == 0) begin
      funct   = SMATMUL_BIAS_THREAD_FUNCT7[6:0];
      op1_col = 1;
      op2_col = 0;
      wr_col  = 0;
      if (rs1[29:10] != 0 || iter != 4 || rs2 != 0 || op1_words != 4)
        `uvm_fatal("CASE", "invalid bias command")
    end else begin
      funct   = SMATMUL_OS_THREAD_FUNCT7[6:0];
      // A and B group g hold K block g; C rows continue across wr_col groups.
      op1_col = cmd.k_groups;
      op2_col = cmd.k_groups;
      wr_col  = cmd.wr_col;
      if (k_groups == 0 || cmd.wr_col == 0) `uvm_fatal("CASE", "SMATMUL_OS needs groups")
      if (op1_bank == op2_bank || op1_bank == wr_bank || op2_bank == wr_bank)
        `uvm_fatal("CASE", "SMATMUL_OS banks must be distinct")
      if (rs2[63:32] != 0 || rs2[23:12] != 16 || iter != 16)
        `uvm_fatal("CASE", "invalid SMATMUL_OS shape or reserved bits")
    end
  endfunction

  function void do_copy(uvm_object rhs);
    smatmul_cmd_item rhs_;
    super.do_copy(rhs);
    if (!$cast(rhs_, rhs)) `uvm_fatal("COPY", "rhs is not smatmul_cmd_item")
    kind = rhs_.kind;
    command_index = rhs_.command_index;
    command_count = rhs_.command_count;
    op1_words = rhs_.op1_words;
    op2_words = rhs_.op2_words;
    k_groups = rhs_.k_groups;
    rs1 = rhs_.rs1;
  endfunction
endclass
