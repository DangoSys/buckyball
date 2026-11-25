class driver extends uvm_driver #(cycle_item);
  `uvm_component_utils(driver)
  virtual bus_if vif;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual bus_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "PRM driver requires an interface")
  endfunction
  task run_phase(uvm_phase phase);
    cycle_item req;
    forever begin
      seq_item_port.get_next_item(req);
      @(negedge vif.clock);
      vif.reset = req.reset;
      foreach (vif.ports[b]) begin
        vif.ports[b].read_valid = req.pins[b].read_valid;
        vif.ports[b].read_addr = req.pins[b].read_addr;
        vif.ports[b].read_result_ready = req.pins[b].read_result_ready;
        vif.ports[b].write_valid = req.pins[b].write_valid;
        vif.ports[b].write_addr = req.pins[b].write_addr;
        vif.ports[b].write_data = req.pins[b].write_data;
        vif.ports[b].write_mask = req.pins[b].write_mask;
        vif.ports[b].write_result_ready = req.pins[b].write_result_ready;
        vif.ports[b].clear = req.pins[b].clear;
        vif.ports[b].bind_generation = req.pins[b].bind_generation;
      end
      @(posedge vif.clock);
      req.sampled = vif.ports;
      #1;
      seq_item_port.item_done();
    end
  endtask
endclass
