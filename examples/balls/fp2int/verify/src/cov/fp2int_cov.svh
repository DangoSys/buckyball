class fp2int_cov extends uvm_component;
  `uvm_component_utils(fp2int_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, fp2int_cov) cmd_imp;

  covergroup cmd_cg;
    coverpoint cur_layout {bins i32 = {0}; bins i8 = {1};}
    coverpoint cur_iter {bins it1 = {1}; bins it4 = {4};}
  endgroup

  int unsigned cur_layout;
  int unsigned cur_iter;
  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    cur_layout = (item.op1_col == 5'd4 && item.wr_col == 5'd1) ? 1 : 0;
    cur_iter   = int'(item.iter);
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0) begin
      `uvm_fatal(
          "COV",
          $sformatf("fp2int_cov coverage incomplete: %0.2f%% (need layout {i32,i8} and iter {1,4})",
                    cmd_cg.get_coverage()))
    end
  endfunction
endclass
