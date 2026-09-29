class env extends checked_env #(request_item, response_item, ref_model);
  `uvm_component_utils(env)
  virtual cache_request_if request;
  virtual cache_response_if response;
  bit block_responses = 0;
  int unsigned accepted = 0;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual cache_request_if)::get(
            this, "", "request", request
        ) || !uvm_config_db#(virtual cache_response_if)::get(
            this, "", "response", response
        ))
      `uvm_fatal("VIF", "Cache interfaces missing")
  endfunction
  task run_phase(uvm_phase phase);
    request_item  req;
    response_item rsp;
    fork
      forever begin
        @(response.cb);
        response.cb.ready <= !response.reset && !block_responses;
      end
      forever begin
        @(posedge request.clock);
        if (request.reset) begin
          scoreboard.cancel_pending();
          cache_ref_reset(model.model);
        end else begin
          if (request.valid && request.ready) begin
            req = request_item::type_id::create("req");
            req.id = request.id;
            req.op = request.op;
            req.addr = request.addr;
            req.way = request.way;
            req.data = request.data;
            req.mask = request.mask;
            req.metadata = request.metadata;
            req.eligible = request.eligible;
            input_export.write(req);
            accepted++;
          end
          if (response.valid && response.ready) begin
            rsp = response_item::type_id::create("rsp");
            rsp.id = response.id;
            rsp.hit = response.hit;
            rsp.available = response.available;
            rsp.entry_valid = response.entry_valid;
            rsp.way = response.way;
            rsp.addr = response.addr;
            rsp.data = response.data;
            rsp.metadata = response.metadata;
            output_export.write(rsp);
          end
        end
      end
    join
  endtask
endclass
