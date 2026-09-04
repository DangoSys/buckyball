class smatmul_cov extends uvm_component;
  `uvm_component_utils(smatmul_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, smatmul_cov) cmd_imp;
  bit saw_m[int];
  bit saw_first;
  bit saw_continuation;
  bit saw_bias;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    int unsigned m;
    if (item.funct7 == SMATMUL_BIAS_CORE_FUNCT7[6:0]) begin
      if (item.rs1[29:10] != 0 || item.iter != 4 || item.rs2 != 0)
        `uvm_fatal("COV", "invalid bias encoding")
      saw_bias = 1'b1;
      return;
    end
    if (item.funct7 != SMATMUL_OS_CORE_FUNCT7[6:0]) `uvm_fatal("COV", "unexpected SMatMul funct7")
    m = int'(item.rs2[11:0]);
    if (item.rs2[23:12] != 16 || item.iter != 16 || item.rs2[63:26] != 0)
      `uvm_fatal("COV", "invalid OS shape or reserved bits")
    if (m != 16) `uvm_fatal("COV", $sformatf("unexpected M=%0d", m))
    saw_m[m] = 1'b1;
    saw_first |= item.rs2[24];
    saw_continuation |= !item.rs2[24];
  endfunction

  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (!saw_bias || !saw_first || !saw_continuation)
      `uvm_fatal("COV", "missing bias, first, or continuation coverage")
    if (!saw_m.exists(16)) `uvm_fatal("COV", "missing M=16 coverage")
  endfunction
endclass
