class endpoint_driver extends uvm_component;
  `uvm_component_utils(endpoint_driver)
  virtual chi_sram_if vif;
  protocol_model state;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual chi_sram_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "CHI endpoint interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(negedge vif.clock);
      vif.rx_active_req = state.idle.rx_active_req;
      vif.tx_active_ack = state.idle.tx_active_ack;
      vif.req_pend = state.idle.req_pend;
      vif.dat_pend = state.idle.dat_pend;
      vif.rsp_credit = !vif.reset && vif.tx_active_req && vif.tx_active_ack && state.grant_enabled && state.rsp_granted < 15 && state.cycle % 5 != 0;
      vif.out_dat_credit = !vif.reset && vif.tx_active_req && vif.tx_active_ack && state.grant_enabled && state.out_dat_granted < 15 && state.cycle % 7 < 3;
    end
  endtask
endclass
