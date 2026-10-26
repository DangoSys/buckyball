class monitor extends uvm_monitor;
  `uvm_component_utils(monitor)
  virtual port_if vif;
  uvm_analysis_port #(request_item) requests;
  uvm_analysis_port #(response_item) responses;
  uvm_analysis_port #(bit) reset_ap;
  function new(string name, uvm_component parent);
    super.new(name, parent);
    requests  = new("requests", this);
    responses = new("responses", this);
    reset_ap  = new("reset_ap", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual port_if)::get(this, "", "vif", vif))
      `uvm_fatal("INTERFACE", "SPM monitor requires ports")
  endfunction
  task run_phase(uvm_phase phase);
    bit pending = 0;
    request_item req;
    response_item rsp;
    forever begin
      @(posedge vif.clock);
      if (vif.reset) begin
        pending = 0;
        reset_ap.write(1);
      end else begin
        if (!pending && vif.busy) `uvm_fatal("BUSY", "SPM is busy without an accepted request")
        if (vif.request_valid && vif.request_ready) begin
          if (pending) `uvm_fatal("OUTSTANDING", "SPM accepted a second request")
          req = request_item::type_id::create("request");
          req.address = vif.address;
          req.size = vif.size;
          req.write = vif.write;
          req.data = vif.data;
          req.mask = vif.mask;
          req.read_only = vif.read_only;
          req.stall = vif.stall;
          req.cancel = vif.cancel;
          requests.write(req);
          pending = 1;
        end else if (pending && (vif.request_ready || !vif.busy))
          `uvm_fatal("BUSY", "SPM reopened its request port before completion")
        if (vif.response_valid && !pending)
          `uvm_fatal("EXTRA_RESULT", "SPM produced an unsolicited result")
        if (vif.response_valid && vif.response_ready) begin
          rsp = response_item::type_id::create("response");
          rsp.value = vif.value;
          rsp.error = vif.error;
          responses.write(rsp);
          pending = 0;
        end
      end
    end
  endtask
endclass
