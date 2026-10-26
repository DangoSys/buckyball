class monitor extends uvm_monitor;
  `uvm_component_utils(monitor)
  virtual chi_retry_if vif;
  uvm_analysis_port #(sample_item) ap;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    ap = new("ap", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual chi_retry_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "CHI monitor interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    sample_item t;
    forever begin
      @(posedge vif.clock);
      t = sample_item::type_id::create("observed");
      t.sampled.reset = vif.reset;
      t.sampled.req_in_valid = vif.req_in_valid;
      t.sampled.req_in_ready = vif.req_in_ready;
      t.sampled.req_out_valid = vif.req_out_valid;
      t.sampled.req_out_ready = vif.req_out_ready;
      t.sampled.req_in = vif.req_in;
      t.sampled.req_out = vif.req_out;
      t.sampled.rsp_in_valid = vif.rsp_in_valid;
      t.sampled.rsp_in_ready = vif.rsp_in_ready;
      t.sampled.rsp_out_valid = vif.rsp_out_valid;
      t.sampled.rsp_out_ready = vif.rsp_out_ready;
      t.sampled.rsp_in = vif.rsp_in;
      t.sampled.rsp_out = vif.rsp_out;
      t.sampled.accepted_valid = vif.accepted_valid;
      t.sampled.accepted_src = vif.accepted_src;
      t.sampled.accepted_txn = vif.accepted_txn;
      t.sampled.pending = vif.pending;
      ap.write(t);
    end
  endtask
endclass
