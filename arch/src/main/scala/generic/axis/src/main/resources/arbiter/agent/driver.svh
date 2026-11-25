class driver extends uvm_driver #(cycle_item);
  `uvm_component_utils(driver)
  virtual arbiter_if vif;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual arbiter_if)::get(this, "", "vif", vif))
      `uvm_fatal("VIF", "Arbiter driver requires vif")
  endfunction

  task run_phase(uvm_phase phase);
    cycle_item req;
    vif.reset = 1;
    vif.in_valid = 0;
    vif.in_last = 0;
    vif.in_data = '{0, 0};
    vif.in_keep = '{0, 0};
    vif.out_ready = 0;
    forever begin
      seq_item_port.get_next_item(req);
      // Drive on falling edges; monitor and DUT sample on rising edges.
      @(negedge vif.clock);
      vif.reset = req.reset;
      vif.in_valid = req.valid;
      vif.in_data = req.data;
      vif.in_keep = req.keep;
      vif.in_last = req.last;
      vif.out_ready = req.ready;
      @(posedge vif.clock);
      seq_item_port.item_done();
    end
  endtask
endclass
