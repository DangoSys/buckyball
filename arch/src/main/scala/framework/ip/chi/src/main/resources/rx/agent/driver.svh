class driver extends uvm_driver #(cycle_item);
  `uvm_component_utils(driver)
  virtual link_if vif;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual link_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "CHI link interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    cycle_item req;
    forever begin
      seq_item_port.get_next_item(req);
      @(negedge vif.clock);
      vif.reset = req.reset;
      vif.pins.active = req.pins.active;
      vif.pins.input_valid = req.pins.input_valid;
      vif.pins.input_data = req.pins.input_data;
      vif.pins.pend = req.pins.pend;
      vif.pins.output_ready = req.pins.output_ready;
      @(posedge vif.clock);
      req.sampled = vif.pins;
      #1;
      seq_item_port.item_done();
    end
  endtask
endclass
