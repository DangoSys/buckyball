class source_driver #(
    int PORT = 0
) extends uvm_driver #(packet_item);
  `uvm_component_param_utils(source_driver#(PORT))
  virtual chi_retry_if vif;
  protocol_model state;
  packet_item pending;
  bit driven = 0;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual chi_retry_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "CHI source interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(negedge vif.clock);
      if (pending == null) seq_item_port.try_next_item(pending);
      driven = 0;
      if (PORT == 0) begin
        vif.req_in = pending != null ? pending.payload : state.idle.req_in;
        vif.req_in_valid = pending != null;
        driven = pending != null;
      end else if (PORT == 1) begin
        vif.rsp_in = pending != null ? pending.payload : state.idle.rsp_in;
        vif.rsp_in_valid = pending != null;
        driven = pending != null;
      end else begin
        vif.accepted_src = pending != null ? pending.payload[18:12] : 0;
        vif.accepted_txn = pending != null ? pending.payload[11:0] : 0;
        vif.accepted_valid = pending != null;
        driven = pending != null;
      end
      @(posedge vif.clock);
      if (!vif.reset && driven && (PORT == 2 || (PORT == 0 ? vif.req_in_ready : vif.rsp_in_ready))) begin
        #1;
        seq_item_port.item_done();
        pending = null;
      end
    end
  endtask
endclass
