class env extends uvm_env;
  `uvm_component_utils(env)
  virtual prepared_map_control_if ctl;
  virtual stream_if #(`PF_RESERVE_WIDTH) reserve;
  virtual stream_if #(`PF_OUT_WIDTH) prepared;
  virtual stream_if #(`PF_MAP_READY_WIDTH) mapped;
  virtual stream_if #(`PF_RELEASE_WIDTH) retire;
  chandle model;
  // A lookup answers in the third cycle of a held query; recent queries and expectations per port.
  logic [`PF_QUERY_WIDTH+`PF_RESULT_WIDTH-1:0] recent[2][$];
  bit notified[int];
  int status[int];
  int
      queries = 0,
      hits = 0,
      misses = 0,
      allocations = 0,
      notifications = 0,
      retired = 0,
      stalls = 0,
      dual = 0,
      cancelled = 0;
  covergroup behavior with function sample (bit hit, bit writing, bit valid, int error);
    option.per_instance = 1;
    hits: coverpoint hit;
    direction: coverpoint writing;
    enabled: coverpoint valid;
    outcomes: coverpoint error {bins all[] = {[0 : 6]};}
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();

  endfunction
  function void verify_contract(bit ok, string message);
    verification_pkg::evidence::checks++;
    if (!ok) `uvm_fatal("PREPARED_MAP", message)
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    verify_contract(uvm_config_db#(virtual prepared_map_control_if)::get(this, "", "ctl", ctl),
                    "ctl missing");
    verify_contract(uvm_config_db#(virtual stream_if #(`PF_RESERVE_WIDTH))::get(
                    this, "", "reserve", reserve), "reserve missing");
    verify_contract(uvm_config_db#(virtual stream_if #(`PF_OUT_WIDTH))::get(
                    this, "", "prepared", prepared), "prepared missing");
    verify_contract(uvm_config_db#(virtual stream_if #(`PF_MAP_READY_WIDTH))::get(
                    this, "", "mapped", mapped), "mapped missing");
    verify_contract(uvm_config_db#(virtual stream_if #(`PF_RELEASE_WIDTH))::get(
                    this, "", "retire", retire), "retire missing");
    model = pmap_ref_create(`PF_ADDRESS_BITS);
  endfunction
  function void query_check(int port, logic [`PF_QUERY_WIDTH-1:0] q,
                            logic [`PF_RESULT_WIDTH-1:0] actual);
    int hit, error;
    u64 pa;
    logic [`PF_RESULT_WIDTH-1:0] expected = '0;
    pmap_ref_query(model, q[`PF(QUERY, VALID)], q[`PF(QUERY, ID)], q[`PF(QUERY, VA)], q[
                   `PF(QUERY, BYTES)], q[`PF(QUERY, WRITE)],
                   retire.sample.valid && retire.sample.ready, retire.sample.bits[`PF(RELEASE, ID)],
                   hit, pa, error);
    expected[`PF(RESULT, HIT)] = 1'(hit);
    expected[`PF(RESULT, PA)] = `PF_ADDRESS_BITS'(pa);
    expected[`PF(RESULT, ERROR)] = 3'(error);
    recent[port].push_back({q, expected});
    if (recent[port].size() > 3) void'(recent[port].pop_front());
    if (recent[port].size() < 3 || recent[port][0] !== recent[port][1] ||
        recent[port][1] !== recent[port][2])
      return;
    verify_contract(actual[`PF(RESULT, HIT)] === 1'(hit) && actual[`PF(RESULT, PA)
                    ] === `PF_ADDRESS_BITS'(pa) && actual[`PF(RESULT, ERROR)] === 3'(error),
                    $sformatf(
                    "query id%0d VA%h bytes%0d expected hit%0d PA%h error%0d got %h",
                    q[
                    `PF(QUERY, ID)
                    ],
                    q[
                    `PF(QUERY, VA)
                    ],
                    q[
                    `PF(QUERY, BYTES)
                    ],
                    hit,
                    pa,
                    error,
                    actual
                    ));
    behavior.sample(hit, q[`PF(QUERY, WRITE)], q[`PF(QUERY, VALID)], error);
    if (q[`PF(QUERY, VALID)]) begin
      queries++;
      if (hit) hits++;
      else misses++;
    end
  endfunction
  function void final_phase(uvm_phase phase);
    pmap_ref_destroy(model);
    super.final_phase(phase);
  endfunction
  `include "agent/service.svh"
  `include "seq/scenario.svh"
endclass
