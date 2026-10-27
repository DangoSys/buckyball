class vector_cov extends uvm_component;
  `uvm_component_utils(vector_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, vector_cov) cmd_imp;

  int unsigned cur_iter;
  int unsigned cur_op1_group;
  int unsigned cur_op2_group;
  int unsigned cur_wr_group;

  covergroup cmd_cg;
    coverpoint cur_iter {bins one = {1}; bins short = {[2 : 16]}; bins deep = {[17 : $]};}
    coverpoint cur_op1_group {bins zero = {0}; bins based = {[1 : $]};}
    coverpoint cur_op2_group {bins zero = {0}; bins based = {[1 : $]};}
    coverpoint cur_wr_group {bins zero = {0}; bins based = {[1 : $]};}
  endgroup

  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    if (item.funct != VECMAT16_THREAD_FUNCT7[6:0])
      `uvm_fatal("COV", $sformatf("unexpected funct %0d", item.funct))
    cur_iter      = item.iter;
    cur_op1_group = item.op1_group;
    cur_op2_group = item.op2_group;
    cur_wr_group  = item.wr_group;
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("vector_cov %0.2f%%", cmd_cg.get_coverage()))
  endfunction
endclass
