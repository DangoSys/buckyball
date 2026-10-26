class command_driver extends uvm_driver #(command);
  `uvm_component_utils(command_driver)
  virtual port_if vif;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual port_if)::get(this, "", "control", vif))
      `uvm_fatal("VIF", "BankSet control interface is required")
  endfunction
  task run_phase(uvm_phase phase);
    command req;
    vif.valid = 0;
    vif.write = 0;
    vif.addr = 0;
    vif.beats = 0;
    vif.done_ready = 0;
    forever begin
      seq_item_port.get_next_item(req);
      do @(negedge vif.clock); while (vif.reset);
      vif.valid = 1;
      vif.write = req.write;
      vif.addr  = req.word_addr * 4;
      vif.beats = req.beats;
      do @(posedge vif.clock); while (!vif.ready);
      @(negedge vif.clock);
      vif.valid = 0;
      wait (vif.done_valid);
      repeat (req.completion_stall) @(negedge vif.clock);
      @(negedge vif.clock);
      vif.done_ready = 1;
      @(posedge vif.clock);
      @(negedge vif.clock);
      vif.done_ready = 0;
      evidence::check(vif.ready === 1'b1 && vif.done_valid === 1'b0,
                      "BankSet did not return to idle after completion");
      seq_item_port.item_done();
    end
  endtask
endclass
