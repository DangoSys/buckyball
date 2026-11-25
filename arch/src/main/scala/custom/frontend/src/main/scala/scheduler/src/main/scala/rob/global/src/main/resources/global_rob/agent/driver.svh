class driver extends uvm_driver #(command_item);
  `uvm_component_utils(driver)
  virtual global_if vif;
  command_item pending;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual global_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Global ROB input interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    forever begin
      @(negedge vif.clock);
      if (pending == null) seq_item_port.try_next_item(pending);
      vif.inputs[`GROB_IN_ALLOC_OFFSET+:`GROB_IN_ALLOC_BITS] = pending!=null ? pending.payload : '0;
      `GI(vif.inputs, ALLOC_VALID) = pending != null;
      @(posedge vif.clock);
      if (!vif.reset && pending != null && `GO(vif.outputs, ALLOC_READY)) begin
        #1;
        seq_item_port.item_done();
        pending = null;
      end
    end
  endtask
endclass
