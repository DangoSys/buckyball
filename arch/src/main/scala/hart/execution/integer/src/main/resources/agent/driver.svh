class driver extends uvm_driver #(request_item);
  `uvm_component_utils(driver)
  virtual port_if vif;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual port_if)::get(this, "", "vif", vif))
      `uvm_fatal("INTERFACE", "Driver interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    request_item req;
    vif.sample = 0;
    vif.a = 0;
    vif.rhs = 0;
    vif.fn = 0;
    vif.dw = 0;
    forever begin
      seq_item_port.get_next_item(req);
      do @(negedge vif.clock); while (vif.reset);
      vif.a = req.a;
      vif.rhs = req.rhs;
      vif.fn = req.fn;
      vif.dw = req.dw;
      vif.sample = 1;
      @(posedge vif.clock);
      seq_item_port.item_done();
      @(negedge vif.clock);
      vif.sample = 0;
    end
  endtask
endclass
