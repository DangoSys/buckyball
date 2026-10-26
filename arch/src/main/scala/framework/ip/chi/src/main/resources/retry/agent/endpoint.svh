class endpoint_driver extends uvm_component;
  `uvm_component_utils(endpoint_driver)
  virtual chi_retry_if vif;
  protocol_model state;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual chi_retry_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "CHI endpoint interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(negedge vif.clock);
      vif.req_out_ready = !vif.reset && state.idle.req_out_ready;
      vif.rsp_out_ready = !vif.reset && state.idle.rsp_out_ready;
    end
  endtask
endclass
