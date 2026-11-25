class source_driver #(
    int PORT = 0
) extends uvm_driver #(packet_item);
  `uvm_component_param_utils(source_driver#(PORT))
  virtual chi_sram_if vif;
  protocol_model state;
  packet_item pending;
  bit driven = 0;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual chi_sram_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "CHI source interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(negedge vif.clock);
      if (pending == null) seq_item_port.try_next_item(pending);
      driven = 0;
      if (PORT == 0) begin
        vif.req_flit = pending != null ? pending.payload : state.idle.req_flit;
        driven = pending != null && !vif.reset && state.req_credits > 0;
        vif.req_valid = driven;
      end else begin
        vif.dat_flit = pending != null ? pending.payload : state.idle.dat_flit;
        driven = pending != null && !vif.reset && state.dat_credits > 0;
        vif.dat_valid = driven;
      end
      @(posedge vif.clock);
      if (driven) begin
        #1;
        seq_item_port.item_done();
        pending = null;
      end
    end
  endtask
endclass
