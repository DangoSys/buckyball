// The counter Ball never uses its bank ports: the common BLink coverage would
// require them to fire, so the test replaces it with this one.
class trace_blink_cov extends bb_blink_cov #(`BB_IN_BW, `BB_OUT_BW);
  `uvm_component_utils(trace_blink_cov)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void check_phase(uvm_phase phase);
    if (cmd_fire_count == 0) `uvm_fatal("COV", "cmd handshake never fired")
    if (resp_fire_count != cmd_fire_count)
      `uvm_fatal("COV", $sformatf("%0d responses for %0d commands", resp_fire_count, cmd_fire_count
                 ))
    for (int i = 0; i < `BB_IN_BW; i++)
    if (read_fire_count[i] != 0) `uvm_fatal("COV", $sformatf("read port %0d fired", i))
    for (int i = 0; i < `BB_OUT_BW; i++)
    if (write_fire_count[i] != 0) `uvm_fatal("COV", $sformatf("write port %0d fired", i))
  endfunction
endclass

class trace_cov extends uvm_component;
  `uvm_component_utils(trace_cov)

  uvm_analysis_imp_cmd #(bb_blink_cmd_item, trace_cov) cmd_imp;

  int unsigned cur_subcmd;
  int unsigned cur_ctr;
  bit cur_payload;

  covergroup cmd_cg;
    coverpoint cur_subcmd {bins start = {0}; bins stop = {1}; bins read = {2};}
    coverpoint cur_ctr {bins ctr[] = {[0 : 15]};}
    coverpoint cur_payload {bins zero = {0}; bins nonzero = {1};}
    cross cur_subcmd, cur_ctr;
    cross cur_subcmd, cur_payload;
  endgroup

  int cmd_count;

  function new(string name, uvm_component parent);
    super.new(name, parent);
    cmd_imp = new("cmd_imp", this);
    cmd_cg  = new();
  endfunction

  function void write_cmd(bb_blink_cmd_item item);
    if (item.funct != BDB_COUNTER_THREAD_FUNCT7[6:0])
      `uvm_fatal("COV", $sformatf("unexpected funct %0d", item.funct))
    cur_subcmd  = item.rs2[3:0];
    cur_ctr     = item.rs2[7:4];
    cur_payload = |item.rs2[63:8];
    cmd_count++;
    cmd_cg.sample();
  endfunction

  function void check_phase(uvm_phase phase);
    verification_pkg::evidence::checks++;
    super.check_phase(phase);
    if (cmd_count == 0) `uvm_fatal("COV", "no cmd observed")
    if (cmd_cg.get_coverage() != 100.0)
      `uvm_fatal("COV", $sformatf("trace_cov %0.2f%%", cmd_cg.get_coverage()))
  endfunction
endclass
