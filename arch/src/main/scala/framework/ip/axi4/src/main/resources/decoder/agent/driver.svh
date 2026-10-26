class driver extends uvm_driver #(cycle_item);
  `uvm_component_utils(driver)
  virtual decoder_if vif;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual decoder_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Decoder interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    cycle_item req;
    forever begin
      seq_item_port.get_next_item(req);
      @(negedge vif.clock);
      vif.reset = req.reset;
      vif.ports[0].aw_valid = req.ports[0].aw_valid;
      vif.ports[0].ar_valid = req.ports[0].ar_valid;
      vif.ports[0].w_valid = req.ports[0].w_valid;
      vif.ports[0].r_ready = req.ports[0].r_ready;
      vif.ports[0].b_ready = req.ports[0].b_ready;
      vif.ports[0].aw = req.ports[0].aw;
      vif.ports[0].ar = req.ports[0].ar;
      vif.ports[0].w = req.ports[0].w;
      for (int p = 1; p < 3; p++) begin
        vif.ports[p].aw_ready = req.ports[p].aw_ready;
        vif.ports[p].ar_ready = req.ports[p].ar_ready;
        vif.ports[p].w_ready = req.ports[p].w_ready;
        vif.ports[p].r_valid = req.ports[p].r_valid;
        vif.ports[p].b_valid = req.ports[p].b_valid;
        vif.ports[p].r = req.ports[p].r;
        vif.ports[p].b = req.ports[p].b;
      end
      @(posedge vif.clock);
      req.ports = vif.ports;
      #1;
      seq_item_port.item_done();
    end
  endtask
endclass
