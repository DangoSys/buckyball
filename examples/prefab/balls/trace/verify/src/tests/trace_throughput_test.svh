// BDB_COUNTER throughput: one START. The instruction carries no workload
// (iter is 0, rs1 is zero), so its largest workload is its only one.
// The command is the first of trace case 0, START of counter 0, so the trace
// hook the RTL calls matches the loaded case.
// Against the ideal outside world of the shared bank model: the response
// accepted at once. The shared cycle checker holds it, like every
// instruction, to the Ball's BEMU cycle model (emu/src/latency.rs). Prints
// the cycles as the result.
class trace_throughput_seq extends uvm_sequence #(bb_blink_cmd_item);
  `uvm_object_utils(trace_throughput_seq)

  function new(string name = "trace_throughput_seq");
    super.new(name);
  endfunction

  task body();
    trace_cmd_item item;
    item = trace_cmd_item::type_id::create("item");
    start_item(item);
    item.load_rust_case(0);
    finish_item(item);
  endtask
endclass

class trace_throughput_test extends uvm_test;
  `uvm_component_utils(trace_throughput_test)

  typedef virtual bb_blink_if #(`BB_IN_BW, `BB_OUT_BW) vif_t;
  vif_t vif;
  bb_blink_env #(`BB_IN_BW, `BB_OUT_BW) env;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    if (!uvm_config_db#(vif_t)::get(this, "", "vif", vif))
      `uvm_fatal("NOVIF", "bb_blink_if not found")
    bb_blink_cov#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(trace_blink_cov::get_type());
    bb_blink_cycles#(`BB_IN_BW, `BB_OUT_BW)::type_id::set_type_override(trace_cycles::get_type());
    env = bb_blink_env#(`BB_IN_BW, `BB_OUT_BW)::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    trace_throughput_seq seq;
    phase.raise_objection(this);
    vif.reset = 1'b1;
    repeat (5) @(posedge vif.clock);
    vif.reset = 1'b0;
    repeat (2) @(posedge vif.clock);
    void'(trace_case_load(0));
    seq = trace_throughput_seq::type_id::create("seq");
    seq.start(env.cmd_agent.seqr);
    @(posedge vif.clock iff (vif.cmd_resp_valid && vif.cmd_resp_ready));
    @(posedge vif.clock);
    `uvm_info("BBPERF", $sformatf("BBPERF BDB_COUNTER %0d %0d", 0,
                                  vif.perf_completed - vif.perf_accepted), UVM_LOW)
    phase.drop_objection(this);
  endtask
endclass
