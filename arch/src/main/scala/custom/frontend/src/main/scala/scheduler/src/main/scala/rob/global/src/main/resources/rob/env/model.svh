class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  localparam int N = `ROB_ENTRIES;
  logic [`ROB_DATA_BITS-1:0] data[N];
  bit valid[N], issued[N], done[N], barrier[N];
  bit [N-1:0] dependencies[N];
  int order[$], tail = 0;
  int state_offsets[N][5] = `ROB_STATE_OFFSETS;
  int ready_offsets[N] = `ROB_READY_OFFSETS;
  uvm_analysis_port #(result_item) expected_ap, actual_ap;
  covergroup behavior with function sample (
      int count, int retired, bit blocked, bit simultaneous, bit commit_barrier
  );
    option.per_instance = 1;
    occupancy: coverpoint count {bins empty = {0}; bins partial = {[1 : N - 1]}; bins full = {N};}
    retirement: coverpoint retired {
      bins none = {0}; bins single = {1}; bins multiple = {[2 : `ROB_RETIRE_WIDTH]};
    }
    source_backpressure: coverpoint blocked;
    issue_complete: coverpoint simultaneous;
    sticky_dependency: coverpoint commit_barrier;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected_ap = new("expected_ap", this);
    actual_ap = new("actual_ap", this);
    behavior = new();
  endfunction
  function void write(cycle_item t);
    result_item expected, actual;
    bit [N-1:0] candidates, retiring, old_barriers;
    int retire_count, in_flight, head, ignored;
    bit admit;
    if (t.reset) begin
      foreach (data[i]) begin
        data[i] = 0;
        valid[i] = 0;
        issued[i] = 0;
        done[i] = 0;
        barrier[i] = 0;
        dependencies[i] = 0;
      end
      order.delete();
      tail = 0;
      return;
    end
    candidates = 0;
    retiring = 0;
    retire_count = 0;
    in_flight = 0;
    old_barriers = 0;
    foreach (barrier[i]) old_barriers[i] = barrier[i];
    for (int age = 0; age < `ROB_RETIRE_WIDTH && age < order.size(); age++) begin
      if (!done[order[age]]) break;
      candidates[order[age]] = 1;
    end
    for (int age = 0; age < `ROB_RETIRE_WIDTH && age < order.size(); age++) begin
      if (!done[order[age]] || !t.inputs[ready_offsets[order[age]]]) break;
      retiring[order[age]] = 1;
      retire_count++;
    end
    head  = order.size() ? order[0] : tail;
    admit = order.size() < N && retiring == 0;
    foreach (valid[i]) if (valid[i] && issued[i] && !done[i]) in_flight++;
    expected = result_item::type_id::create("expected");
    actual = result_item::type_id::create("actual");
    expected.payload = '0;
    actual.payload = t.outputs;
    expected.payload[`ROB_OUT_ALLOC_READY_OFFSET+:`ROB_OUT_ALLOC_READY_WIDTH] = admit;
    expected.payload[`ROB_OUT_RETIRECANDIDATES_OFFSET+:`ROB_OUT_RETIRECANDIDATES_WIDTH] = candidates;
    expected.payload[`ROB_OUT_RETIRED_OFFSET+:`ROB_OUT_RETIRED_WIDTH] = retiring;
    expected.payload[`ROB_OUT_HEAD_OFFSET+:`ROB_OUT_HEAD_WIDTH] = head;
    expected.payload[`ROB_OUT_TAIL_OFFSET+:`ROB_OUT_TAIL_WIDTH] = tail;
    expected.payload[`ROB_OUT_ISSUEDCOUNT_OFFSET+:`ROB_OUT_ISSUEDCOUNT_WIDTH] = in_flight;
    expected.payload[`ROB_OUT_EMPTY_OFFSET+:`ROB_OUT_EMPTY_WIDTH] = order.size() == 0;
    expected.payload[`ROB_OUT_FULL_OFFSET+:`ROB_OUT_FULL_WIDTH] = order.size() == N;
    foreach (data[i]) begin
      expected.payload[state_offsets[i][0]+:`ROB_DATA_BITS] = data[i];
      expected.payload[state_offsets[i][1]] = valid[i];
      expected.payload[state_offsets[i][2]] = issued[i];
      expected.payload[state_offsets[i][3]] = done[i];
      expected.payload[state_offsets[i][4]+:N] = dependencies[i];
    end
    behavior.sample(order.size(), retire_count, `RI(t.inputs, ALLOC_VALID) && !admit,
                    `RI(t.inputs, ISSUE_VALID) && `RI(t.inputs, COMPLETE_VALID),
                    `RI(t.inputs, ALLOC_BITS_COMMITBARRIER));
    expected_ap.write(expected);
    actual_ap.write(actual);
    if (`RI(t.inputs, ALLOC_VALID) && admit) begin
      data[tail] = `RI(t.inputs, ALLOC_BITS_DATA);
      valid[tail] = 1;
      issued[tail] = 0;
      done[tail] = 0;
      barrier[tail] = `RI(t.inputs, ALLOC_BITS_COMMITBARRIER);
      dependencies[tail] = `RI(t.inputs, ALLOC_BITS_DEPENDENCIES);
      order.push_back(tail);
      tail = (tail + 1) % N;
    end
    if (`RI(t.inputs, ISSUE_VALID)) issued[`RI(t.inputs, ISSUE_BITS)] = 1;
    if (`RI(t.inputs, COMPLETE_VALID)) done[`RI(t.inputs, COMPLETE_BITS)] = 1;
    foreach (dependencies[i]) begin
      dependencies[i] &= ~retiring;
      if (
          `RI(t.inputs, COMPLETE_VALID)
          && !barrier[i] && !old_barriers[
          `RI(t.inputs, COMPLETE_BITS)
          ])
        dependencies[i][`RI(t.inputs, COMPLETE_BITS)] = 0;
    end
    repeat (retire_count) begin
      ignored = order.pop_front();
      valid[ignored] = 0;
      issued[ignored] = 0;
      done[ignored] = 0;
    end
  endfunction
endclass
