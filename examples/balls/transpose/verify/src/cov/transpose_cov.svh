class transpose_cov extends uvm_component;
  `uvm_component_utils(transpose_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, transpose_cov) cmd_imp;

  covergroup cmd_cg;
    coverpoint cur_elem_bits {bins i8 = {8}; bins i32 = {32};}
    coverpoint cur_iter {
      bins it1 = {1}; bins it2 = {2}; bins it4 = {4}; bins it8 = {8}; bins it16 = {16};
    }
  endgroup

  int unsigned cur_elem_bits;
  int unsigned cur_iter;
  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    cur_elem_bits = int'(item.rs2[7:0]);
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
              "transpose_cov coverage incomplete: %0.2f%% (need elem_bits {8,32} and iter {1,2,4,8,16})",
              cmd_cg.get_coverage()))
    end
  endfunction
endclass
