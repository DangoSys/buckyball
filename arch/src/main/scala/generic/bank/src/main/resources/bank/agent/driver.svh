class driver extends uvm_driver #(request_item);
  `uvm_component_utils(driver)
  virtual port_if request;
  virtual port_if response;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual port_if)::get(
            this, "", "request", request
        ) || !uvm_config_db#(virtual port_if)::get(
            this, "", "response", response
        ))
      `uvm_fatal("VIF", "Bank driver interfaces missing")
  endfunction

  task run_phase(uvm_phase phase);
    request_item req;
    request.valid = 0;
    request.write = 0;
    request.addr = 0;
    request.data = 0;
    request.mask = 0;
    request.tag = 0;
    request.reset = 1;
    response.ready = 0;
    forever begin
      seq_item_port.get_next_item(req);
      if (req.reset_command) begin
        @(negedge request.clock);
        request.reset = 1;
        repeat (4) @(negedge request.clock);
        request.reset = 0;
      end else begin
        @(negedge request.clock);
        response.ready = 0;
        request.valid = 1;
        request.write = req.write;
        request.addr = req.addr;
        request.data = req.data;
        request.mask = req.mask;
        request.tag = req.tag;
        do @(posedge request.clock); while (!request.ready);
        @(negedge request.clock);
        request.valid = 0;
        if (req.cancel_response) begin
          request.reset = 1;
          repeat (2) @(negedge request.clock);
          request.reset = 0;
          repeat (3) @(negedge request.clock);
        end else begin
          while (!response.valid) @(negedge request.clock);
          repeat (req.stall_cycles + 1) @(negedge request.clock);
          response.ready = 1;
          @(negedge request.clock);
          // Keep the sink ready while idle; the next transfer selects its stall.
        end
      end
      seq_item_port.item_done();
    end
  endtask
endclass
