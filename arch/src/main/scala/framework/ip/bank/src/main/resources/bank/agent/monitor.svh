class monitor extends uvm_monitor;
  `uvm_component_utils(monitor)
  virtual port_if request;
  virtual port_if response;
  uvm_analysis_port #(request_item) request_ap;
  uvm_analysis_port #(response_item) response_ap;
  uvm_analysis_port #(bit) reset_ap;
  int unsigned accepted = 0;
  int unsigned retired = 0;
  int unsigned cancelled = 0;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    request_ap = new("request_ap", this);
    response_ap = new("response_ap", this);
    reset_ap = new("reset_ap", this);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual port_if)::get(
            this, "", "request", request
        ) || !uvm_config_db#(virtual port_if)::get(
            this, "", "response", response
        ))
      `uvm_fatal("VIF", "Bank monitor interfaces missing")
  endfunction

  task run_phase(uvm_phase phase);
    request_item req;
    response_item rsp;
    bit pending = 0;
    forever begin
      @(posedge request.clock);
      if (request.reset) begin
        if (pending) cancelled++;
        pending = 0;
        reset_ap.write(1);
      end else begin
        if ($isunknown({request.valid, request.ready, response.valid, response.ready}))
          `uvm_fatal("BANK_UNKNOWN", "unknown Bank handshake")
        if (request.valid && request.ready) begin
          if (pending) `uvm_fatal("BANK_PENDING", "accepted a second outstanding request")
          if ($isunknown({request.addr, request.write, request.data, request.mask, request.tag}))
            `uvm_fatal("BANK_UNKNOWN", "unknown accepted request")
          req = request_item::type_id::create("request");
          req.addr = request.addr;
          req.write = request.write;
          req.data = request.data;
          req.mask = request.mask;
          req.tag = request.tag;
          pending = 1;
          accepted++;
          request_ap.write(req);
        end
        if (response.valid && !pending)
          `uvm_fatal("BANK_EXTRA", "response without an outstanding request")
        if (pending && !(request.valid && request.ready) && request.ready !== 1'b0)
          `uvm_fatal("BANK_PENDING", "request ready with an outstanding response")
        if (response.valid && response.ready) begin
          if ($isunknown({response.data, response.tag, response.error}))
            `uvm_fatal("BANK_UNKNOWN", "unknown accepted response")
          rsp = response_item::type_id::create("response");
          rsp.data = response.data;
          rsp.tag = response.tag;
          rsp.error = response.error;
          response_ap.write(rsp);
          pending = 0;
          retired++;
        end
      end
    end
  endtask
endclass
