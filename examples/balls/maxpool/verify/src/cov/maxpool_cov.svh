class maxpool_cov extends uvm_component;
  `uvm_component_utils(maxpool_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, maxpool_cov) cmd_imp;

  int unsigned cur_kernel;
  int unsigned cur_stride;
  int unsigned cur_padding;
  int cmd_count;

  covergroup cmd_cg;
    coverpoint cur_kernel {bins k2 = {2}; bins k3 = {3};}
    coverpoint cur_stride {bins s1 = {1}; bins s2 = {2};}
    coverpoint cur_padding {bins p0 = {0}; bins p1 = {1};}
  endgroup

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    if (item.funct7 != MAXPOOL_CORE_FUNCT7[6:0])
      `uvm_fatal("COV", $sformatf("unexpected funct7 %0d", item.funct7))
    cur_kernel  = int'(item.rs2[11:8]);
    cur_stride  = int'(item.rs2[15:12]);
    cur_padding = int'(item.rs2[19:16]);
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
              "maxpool_cov coverage incomplete: %0.2f%% (need kernel {2,3}, stride {1,2}, pad {0,1})",
              cmd_cg.get_coverage()))
    end
  endfunction
endclass
