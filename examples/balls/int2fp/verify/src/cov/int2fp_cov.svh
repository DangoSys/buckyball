class int2fp_cov extends uvm_component;
  `uvm_component_utils(int2fp_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, int2fp_cov) cmd_imp;

  covergroup cmd_cg;
    coverpoint cur_mode {bins fp32 = {0}; bins i8 = {1};}
    coverpoint cur_iter {bins it1 = {1}; bins it2 = {2}; bins it4 = {4}; bins it16 = {16};}
  endgroup

  int unsigned cur_mode;
  int unsigned cur_iter;
  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    cur_mode = int'(item.special[33:32]);
    cur_iter = int'(item.iter);
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0) begin
      `uvm_fatal(
          "COV",
          $sformatf(
              "int2fp_cov coverage incomplete: %0.2f%% (need mode {fp32,i8} and iter {1,2,4,16})",
              cmd_cg.get_coverage()))
    end
  endfunction
endclass
