class smatmul_cov extends uvm_component;
  `uvm_component_utils(smatmul_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, smatmul_cov) cmd_imp;

  bit saw_m[int];
  bit saw_n[int];
  bit saw_k[int];
  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    int unsigned m;
    int unsigned n;
    int unsigned k;
    m = int'(item.rs2[11:0]);
    n = int'(item.rs2[23:12]);
    k = int'(item.rs2[35:24]);
    if (item.rs2[63:36] != 0) `uvm_fatal("COV", "rs2[63:36] must be 0")
    if (m != 16 && m != 32) `uvm_fatal("COV", $sformatf("rows must be 16 or 32: %0d", m))
    if (n != 16 && n != 32) `uvm_fatal("COV", $sformatf("cols must be 16 or 32: %0d", n))
    if (k != 16) `uvm_fatal("COV", $sformatf("reduction must be 16: %0d", k))
    saw_m[m] = 1'b1;
    saw_n[n] = 1'b1;
    saw_k[k] = 1'b1;
    cmd_count++;
  endfunction

  function void check_phase(uvm_phase phase);
    int unsigned need[2] = '{16, 32};
    int i;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    for (i = 0; i < 2; i++) begin
      if (!saw_m.exists(need[i]) || !saw_m[need[i]])
        `uvm_fatal("COV", $sformatf("missing M bin %0d", need[i]))
      if (!saw_n.exists(need[i]) || !saw_n[need[i]])
        `uvm_fatal("COV", $sformatf("missing N bin %0d", need[i]))
    end
    if (!saw_k.exists(16) || !saw_k[16]) `uvm_fatal("COV", "missing K bin 16")
  endfunction
endclass
