class relu_cov extends uvm_component;
  `uvm_component_utils(relu_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, relu_cov) cmd_imp;

  int unsigned bank_entries;
  int unsigned cur_segments;
  int unsigned cur_group;
  int unsigned cur_group_base;

  covergroup cmd_cg;
    // One segment, several segments, every row of the bank in segments of 16.
    coverpoint cur_segments {
      bins one = {1}; bins some = {[2 : 63]}; bins many = {[64 : $]};
    }
    coverpoint cur_group {bins first = {0}; bins later = {[1 : $]};}
    coverpoint cur_group_base {bins zero = {0}; bins offset = {[1 : $]};}
  endgroup

  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg = new();
    bank_entries = relu_bank_entries();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    if (item.funct != RELU_THREAD_FUNCT7[6:0])
      `uvm_fatal("COV", $sformatf("unexpected funct %0d", item.funct))
    if (item.rs2 == 0) `uvm_fatal("COV", "zero stride")
    cur_segments   = bank_entries / int'(item.rs2);
    cur_group      = item.op2_bank;
    cur_group_base = item.op1_group;
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("relu_cov %0.2f%%", cmd_cg.get_coverage()))
  endfunction
endclass
