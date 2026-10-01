class env extends checked_env #(request_item, response_item, ref_model);
  `uvm_component_utils(env)
  virtual bank_if request;
  virtual bank_if response;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual bank_if)::get(
            this, "", "request", request
        ) || !uvm_config_db#(virtual bank_if)::get(
            this, "", "response", response
        ))
      `uvm_fatal("VIF", "bank interfaces missing")
  endfunction

  task run_phase(uvm_phase phase);
    request_item  req;
    response_item rsp;
    forever begin
      @(posedge request.clock);
      if (request.reset) begin
        scoreboard.cancel_pending();
      end else begin
        if (request.valid && request.ready) begin
          req = request_item::type_id::create("request");
          req.addr = request.addr;
          req.write = request.write;
          req.data = request.data;
          req.mask = request.mask;
          req.tag = request.tag;
          input_export.write(req);
        end
        if (response.valid && response.ready) begin
          rsp = response_item::type_id::create("response");
          rsp.data = response.data;
          rsp.tag = response.tag;
          rsp.error = response.error;
          output_export.write(rsp);
        end
      end
    end
  endtask
endclass
