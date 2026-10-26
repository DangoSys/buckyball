class driver extends uvm_driver #(request_item);
  `uvm_component_utils(driver)
  virtual port_if  vif;
  reset_controller resetter;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual port_if)::get(
            this, "", "vif", vif
        ) || !uvm_config_db#(reset_controller)::get(
            this, "", "resetter", resetter
        ))
      `uvm_fatal("INTERFACE", "FPU driver requires ports and reset controller")
  endfunction
  task run_phase(uvm_phase phase);
    request_item req;
    vif.request_valid = 0;
    vif.response_ready = 0;
    vif.instruction = 0;
    vif.integer_value = 0;
    vif.a = 0;
    vif.rhs = 0;
    vif.c = 0;
    vif.rounding = 0;
    forever begin
      seq_item_port.get_next_item(req);
      do @(negedge vif.clock); while (vif.reset);
      vif.instruction = req.instruction;
      vif.integer_value = req.integer_value;
      vif.a = req.a;
      vif.rhs = req.rhs;
      vif.c = req.c;
      vif.rounding = req.rounding;
      vif.stall = req.stall;
      vif.cancel = req.cancel;
      vif.request_valid = 1;
      do @(posedge vif.clock); while (!vif.request_ready);
      @(negedge vif.clock);
      vif.request_valid = 0;
      if (req.cancel) begin
        resetter.pulse(2);
      end else begin
        while (!vif.response_valid) @(negedge vif.clock);
        repeat (req.stall) @(negedge vif.clock);
        vif.response_ready = 1;
        @(posedge vif.clock);
        @(negedge vif.clock);
        vif.response_ready = 0;
      end
      seq_item_port.item_done();
    end
  endtask
endclass
