class driver extends uvm_driver #(cycle_item);
  `uvm_component_utils(driver)
  virtual admission_if vif;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual admission_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Admission interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    cycle_item req;
    forever begin
      seq_item_port.get_next_item(req);
      @(negedge vif.clock);
      vif.reset  = req.reset;
      vif.inputs = req.inputs;
      @(posedge vif.clock);
      req.outputs = vif.outputs;
      #1;
      seq_item_port.item_done();
    end
  endtask
endclass
