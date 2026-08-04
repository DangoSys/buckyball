class im2col_cov extends uvm_component;
  `uvm_component_utils(im2col_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, im2col_cov) cmd_imp;

  covergroup cmd_cg;
    coverpoint cur_ksize {bins k1 = {1}; bins k3 = {3}; bins k5 = {5};}
    coverpoint cur_stride {bins s1 = {1};}
    coverpoint cur_pad {bins p0 = {0}; bins p1 = {1};}
    coverpoint cur_iter {
      bins it3 = {3}; bins it4 = {4}; bins it5 = {5}; bins it6 = {6}; bins it7 = {7};
    }
  endgroup

  int unsigned cur_ksize;
  int unsigned cur_stride;
  int unsigned cur_pad;
  int unsigned cur_iter;
  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    cur_ksize = int'(item.special[7:0]);
    cur_stride = int'(item.special[15:8]);
    cur_pad = int'(item.special[23:16]);
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
              "im2col_cov coverage incomplete: %0.2f%% (need ksize {1,3,5}, stride {1}, pad {0,1}, iter {3,4,5,6,7})",
              cmd_cg.get_coverage()))
    end
  endfunction
endclass
