class checker extends uvm_component;
  `uvm_component_utils(checker)

  virtual arbiter_if vif;
  chandle model;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual arbiter_if)::get(this, "", "vif", vif)) begin
      `uvm_fatal("VIF", "checker requires vif")
    end
    model = axis_arbiter_create();
    if (model == null) `uvm_fatal("MODEL", "failed to create arbiter model")
  endfunction

  task run_phase(uvm_phase phase);
    int unsigned selected;
    bit expected_valid;
    bit [1:0] expected_ready;

    forever begin
      @(posedge vif.clock);
      if (!vif.reset) begin
        selected = axis_arbiter_step(model, vif.in_valid, vif.out_ready, vif.in_last);
        expected_valid = selected != 32'hffffffff;
        expected_ready = expected_valid && vif.out_ready ? 2'b01 << selected : 2'b00;

        if (vif.out_valid !== expected_valid) `uvm_fatal("VALID", "output valid mismatch")
        if (vif.in_ready !== expected_ready) `uvm_fatal("READY", "input ready mismatch")
        if (expected_valid) begin
          if (vif.out_data !== vif.in_data[selected]) `uvm_fatal("DATA", "selected data mismatch")
          if (vif.out_keep !== vif.in_keep[selected]) `uvm_fatal("KEEP", "selected keep mismatch")
          if (vif.out_last !== vif.in_last[selected]) `uvm_fatal("LAST", "selected last mismatch")
        end
      end
    end
  endtask

  function void final_phase(uvm_phase phase);
    axis_arbiter_destroy(model);
    super.final_phase(phase);
  endfunction
endclass
