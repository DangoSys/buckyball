class protocol_model extends uvm_subscriber #(sample_item);
  `uvm_component_utils(protocol_model)
  chandle reference;
  int req_credits = 0, dat_credits = 0, rsp_granted = 0, out_dat_granted = 0;
  int cycle = 0, checks = 0;
  localparam int STALL_READS = 15 / `CHI_BEATS + 8;
  bit grant_enabled = 1;
  rsp_t responses[$];
  dat_t data_responses[$];
  string scenario;
  pins_t sampled;
  pins_t idle;
  uvm_analysis_port #(result_item) expected_ap, actual_ap;
  uvm_analysis_port #(bit) reset_ap;
  covergroup behavior with function sample (int op, bit error);
    option.per_instance = 1;
    request_opcode: coverpoint op {
      bins read = {4}; bins write_full = {'h1d}; bins write_partial = {'h1c};
    }
    response_error: coverpoint error;
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
    reference = chi_ref_create(256);
  endfunction
  function void write(sample_item t);
    pins_t s = t.sampled;
    sampled = s;
    if (s.req_valid) begin
      req_t req = req_t'(s.req_flit);
      behavior.sample(req.opcode, 0);
    end
    if (s.rsp_valid) begin
      rsp_t rsp = rsp_t'(s.rsp_flit);
      behavior.sample(4, rsp.error != 0);
    end
    if (s.out_dat_valid) begin
      dat_t dat = dat_t'(s.out_dat_flit);
      behavior.sample(4, dat.error != 0);
    end

    if (s.reset) begin
      req_credits = 0;
      dat_credits = 0;
      rsp_granted = 0;
      out_dat_granted = 0;
      responses.delete();
      data_responses.delete();
    end else begin
      cycle++;
      req_credits += int'(s.req_credit) - int'(s.req_valid);
      dat_credits += int'(s.dat_credit) - int'(s.dat_valid);
      rsp_granted += int'(s.rsp_credit) - int'(s.rsp_valid);
      out_dat_granted += int'(s.out_dat_credit) - int'(s.out_dat_valid);
      if (req_credits < 0 || dat_credits < 0 || rsp_granted < 0 || out_dat_granted < 0)
        `uvm_fatal("CREDIT", "Flit without credit")
      if (s.rsp_valid) responses.push_back(rsp_t'(s.rsp_flit));
      if (s.out_dat_valid) data_responses.push_back(dat_t'(s.out_dat_flit));
    end

  endfunction
  function void final_phase(uvm_phase phase);
    chi_ref_destroy(reference);
    super.final_phase(phase);
  endfunction
  function void check_phase(uvm_phase phase);
    evidence::check(responses.size() == 0 && data_responses.size() == 0,
                    "CHI SRAM responses remain unchecked");
  endfunction
endclass
