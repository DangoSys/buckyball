class protocol_model extends uvm_subscriber #(sample_item);
  `uvm_component_utils(protocol_model)
  req_t requests[$];
  rsp_t responses[$], original_responses[$];
  int checks = 0;
  req_t originals[int];
  int retry_types[int];
  string scenario;
  pins_t sampled;
  pins_t idle;
  uvm_analysis_port #(result_item) expected_ap, actual_ap;
  uvm_analysis_port #(bit) reset_ap;
  covergroup behavior with function sample (int count, bit stalled, bit reissued);
    option.per_instance = 1;
    live: coverpoint count {
      bins empty = {0}; bins partial = {[1 : `CHI_RECORDS - 1]}; bins full = {`CHI_RECORDS};
    }
    backpressure: coverpoint stalled;
    retry_request: coverpoint reissued;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();
    expected_ap = new("expected_ap", this);
    actual_ap = new("actual_ap", this);
    reset_ap = new("reset_ap", this);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!$value$plusargs("CASE=%s", scenario)) `uvm_fatal("CASE", "CHI scenario is required")
    idle = '0;

  endfunction
  function void write(sample_item t);
    pins_t s = t.sampled;
    sampled = s;
    begin
      req_t out_req = req_t'(s.req_out);
      behavior.sample(s.pending, s.req_out_valid && !s.req_out_ready,
                      s.req_out_valid && !out_req.allow_retry);
    end

    if (s.reset) begin
      requests.delete();
      responses.delete();
      original_responses.delete();
    end else begin
      if (s.req_in_valid && s.req_in_ready) begin
        req_t r = req_t'(s.req_in);
        originals[r.txn] = r;
      end
      if (s.rsp_in_valid && s.rsp_in_ready) begin
        rsp_t r = rsp_t'(s.rsp_in);
        if (r.opcode == 3) retry_types[r.txn] = r.pcrd_type;
        if (r.opcode != 3 && r.opcode != 7) original_responses.push_back(r);
      end
      if (s.req_out_valid && s.req_out_ready) requests.push_back(req_t'(s.req_out));
      if (s.rsp_out_valid && s.rsp_out_ready) responses.push_back(rsp_t'(s.rsp_out));
    end

  endfunction
  function void check_phase(uvm_phase phase);
    evidence::check(requests.size() == 0 && responses.size() == 0 && original_responses.size() == 0,
                    "CHI retry queues did not drain");
  endfunction
endclass
