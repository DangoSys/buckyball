class monitor extends uvm_monitor;
  `uvm_component_utils(monitor)
  virtual cache_request_if request;
  virtual cache_response_if response;
  uvm_analysis_port #(request_item) request_ap;
  uvm_analysis_port #(response_item) response_ap;
  uvm_analysis_port #(bit) reset_ap;
  int unsigned accepted = 0, retired = 0;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    request_ap = new("request_ap", this);
    response_ap = new("response_ap", this);
    reset_ap = new("reset_ap", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual cache_request_if)::get(
            this, "", "request", request
        ) || !uvm_config_db#(virtual cache_response_if)::get(
            this, "", "response", response
        ))
      `uvm_fatal("VIF", "Cache monitor interfaces are required")
  endfunction
  task run_phase(uvm_phase phase);
    request_item  req;
    response_item rsp;
    forever begin
      @(posedge request.clock);
      if (request.reset) reset_ap.write(1);
      else begin
        if (request.valid && request.ready) begin
          req = request_item::type_id::create("request");
          req.id = request.id;
          req.op = request.op;
          req.addr = request.addr;
          req.way = request.way;
          req.data = request.data;
          req.mask = request.mask;
          req.metadata = request.metadata;
          req.eligible = request.eligible;
          request_ap.write(req);
          accepted++;
        end
        if (response.valid && response.ready) begin
          rsp = response_item::type_id::create("response");
          rsp.id = response.id;
          rsp.hit = response.hit;
          rsp.available = response.available;
          rsp.entry_valid = response.entry_valid;
          rsp.way = response.way;
          rsp.addr = response.addr;
          rsp.data = response.data;
          rsp.metadata = response.metadata;
          response_ap.write(rsp);
          retired++;
        end
      end
    end
  endtask
endclass
