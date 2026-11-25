class cmo_result extends uvm_sequence_item;
  `uvm_object_utils_begin(cmo_result)
    `uvm_field_int(tag, UVM_ALL_ON)
    `uvm_field_int(ok, UVM_ALL_ON)
  `uvm_object_utils_end
  int tag;
  bit ok;
  function new(string name = "result");
    super.new(name);
  endfunction
endclass
class cache_model extends protocol_model;
  `uvm_component_utils(cache_model)
  virtual cmo_if cmo;
  result_board #(cmo_result, 0) acknowledgments;
  bit maintenance_pending;
  longint unsigned next_line, last_line;
  int operation, maintenance_tag, issued, returned, fail_index = -1;
  covergroup maintenance with function sample (int op, bit ok, bit stalled, bit overlap);
    option.per_instance = 1;
    operations: coverpoint op {bins all[] = {[0 : 2]};}
    errors: coverpoint ok;
    backpressure: coverpoint stalled;
    concurrency: coverpoint overlap;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    maintenance = new();
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(virtual cmo_if)::get(this, "", "cmo", cmo))
      `uvm_fatal("VIF", "Cache maintenance interface is required")
    acknowledgments = result_board#(cmo_result, 0)::type_id::create("maintenance_results", this);
  endfunction
  function void write(sample_item t);
    sample_item demand;
    rsp_t response, received_rsp;
    req_t request;
    demand = sample_item::type_id::create("demand");
    demand.sampled = t.sampled;
    if (t.sampled.reset) begin
      acknowledgments.cancel_pending();
      maintenance_pending = 0;
      issued = 0;
      returned = 0;
    end else begin
      if (cmo.valid && cmo.ready) begin
        cmo_result result = cmo_result::type_id::create("expected");
        if (maintenance_pending) `uvm_fatal("CMO", "Overlapping maintenance admissions")
        maintenance_pending = 1;
        next_line = cmo.first_line;
        last_line = cmo.last_line;
        operation = cmo.operation;
        maintenance_tag = cmo.tag;
        issued = 0;
        returned = 0;
        result.tag = maintenance_tag;
        result.ok = fail_index < 0 || fail_index > int'((last_line - next_line) / 64);
        acknowledgments.write_expected(result);
        evidence::check(cmo.older_drained && t.sampled.outstanding == 0 && !t.sampled.access_ready,
                        "Maintenance admitted before drain or alongside a CPU request");
      end
      if (t.sampled.req_valid && t.sampled.req_ready) begin
        request = req_t'(t.sampled.req);
        if (request.txn == `CHI_BANKS) begin
          int opcode=operation==0?`CMO_CLEAN:operation==1?`CMO_CLEAN_INVALID:`CMO_INVALID;
          evidence::check(
              maintenance_pending&&request.addr==next_line&&request.opcode==opcode&&request.src==1&&request.tgt==64&&request.size==6&&request.snp_attr&&request.mem_attr==12&&request.allow_retry,
              "Incorrect maintenance request footprint or attributes");
          response = '0;
          response.src = 64;
          response.tgt = 1;
          response.txn = `CHI_BANKS;
          response.opcode = `CMO_COMP;
          response.error = issued == fail_index ? 2 : 0;
          if (scenario == "bad_maintenance_identity") response.src = 65;
          if (scenario == "bad_maintenance_opcode") response.opcode = 1;
          rx_rsp_queue.push_back(response);
          issued++;
          demand.sampled.req_valid = 0;
        end
      end
      received_rsp = rsp_t'(t.sampled.rx_rsp);
      if (t.sampled.rx_rsp_valid && received_rsp.txn == `CHI_BANKS) begin
        response = rsp_t'(t.sampled.rx_rsp);
        if (t.sampled.rx_rsp_ready) begin
          rsp_active = 0;
          returned++;
          if (response.error == 0) next_line += 64;
        end
        demand.sampled.rx_rsp_valid = 0;
        demand.sampled.rx_rsp_ready = 0;
      end
      if (cmo.response_valid && cmo.response_ready) begin
        cmo_result result = cmo_result::type_id::create("actual");
        evidence::check(
            maintenance_pending && issued == returned && (next_line > last_line || fail_index >= 0),
            "Maintenance acknowledged before all completions");
        result.tag = cmo.response_tag;
        result.ok  = cmo.ok;
        acknowledgments.write_actual(result);
        maintenance_pending = 0;
        maintenance.sample(operation, cmo.ok, 0, t.sampled.outstanding > 0);
      end
      if (cmo.valid && !cmo.ready || cmo.response_valid && !cmo.response_ready)
        maintenance.sample(cmo.operation, 1, 1, t.sampled.outstanding > 0);
      if (maintenance_pending && t.sampled.access_valid) maintenance.sample(operation, 1, 0, 1);
    end
    super.write(demand);
  endfunction
  function void check_phase(uvm_phase phase);
    super.check_phase(phase);
    if (maintenance_pending) `uvm_fatal("CMO", "Maintenance did not drain")
  endfunction
endclass
