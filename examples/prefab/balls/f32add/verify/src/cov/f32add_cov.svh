class f32add_cov extends uvm_component;
  `uvm_component_utils(f32add_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, f32add_cov) cmd_imp;

  bit cur_first;
  int unsigned cur_group;
  int unsigned cur_rows;

  covergroup cmd_cg;
    coverpoint cur_first {bins accumulate = {0}; bins first = {1};}
    coverpoint cur_group {bins zero = {0}; bins later = {[1 : 31]};}
    coverpoint cur_rows {bins one_group = {[1 : 16]}; bins deep = {[17 : $]};}
    cross cur_first, cur_group;
  endgroup

  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    if (item.funct != F32ADD_THREAD_FUNCT7[6:0])
      `uvm_fatal("COV", $sformatf("unexpected funct %0d", item.funct))
    cur_first = item.rs2[5];
    cur_group = item.rs2[4:0];
    cur_rows  = item.iter;
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("f32add_cov %0.2f%%", cmd_cg.get_coverage()))
  endfunction
endclass
