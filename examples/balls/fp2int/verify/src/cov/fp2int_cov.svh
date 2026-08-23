class fp2int_cov extends uvm_component;
  `uvm_component_utils(fp2int_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, fp2int_cov) cmd_imp;

  covergroup cmd_cg;
    coverpoint cur_source_groups {bins g1 = {1}; bins g2 = {2}; bins g4 = {4}; bins g8 = {8};}
    coverpoint cur_destination_groups {bins g1 = {1};}
    coverpoint cur_iter {bins it1 = {1};}
  endgroup

  int unsigned cur_source_groups;
  int unsigned cur_destination_groups;
  int unsigned cur_iter;
  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    cur_source_groups = item.op1_col;
    cur_destination_groups = item.wr_col;
    cur_iter = int'(item.iter);
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
  endfunction
endclass
