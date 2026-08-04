class matrix_cov extends uvm_component;
  `uvm_component_utils(matrix_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, matrix_cov) cmd_imp;

  bit saw_os;
  bit saw_ws;
  bit saw_m[int];
  bit saw_n[int];
  bit saw_k[int];
  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    int unsigned mode;
    int unsigned m;
    int unsigned n;
    int unsigned k;
    mode = int'(item.rs2[36]);
    m = int'(item.rs2[11:0]);
    n = int'(item.rs2[23:12]);
    k = int'(item.rs2[35:24]);
    if (mode == 0) saw_os = 1'b1;
    if (mode == 1) saw_ws = 1'b1;
    saw_m[m] = 1'b1;
    saw_n[n] = 1'b1;
    saw_k[k] = 1'b1;
    cmd_count++;
  endfunction

  function void check_phase(uvm_phase phase);
    int unsigned need[5] = '{1, 2, 4, 8, 16};
    int i;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (!saw_os || !saw_ws) `uvm_fatal("COV", "mode bins incomplete (need OS and WS)")
    for (i = 0; i < 5; i++) begin
      if (!saw_m.exists(need[i]) || !saw_m[need[i]])
        `uvm_fatal("COV", $sformatf("missing M bin %0d", need[i]))
      if (!saw_n.exists(need[i]) || !saw_n[need[i]])
        `uvm_fatal("COV", $sformatf("missing N bin %0d", need[i]))
      if (!saw_k.exists(need[i]) || !saw_k[need[i]])
        `uvm_fatal("COV", $sformatf("missing K bin %0d", need[i]))
    end
  endfunction
endclass
