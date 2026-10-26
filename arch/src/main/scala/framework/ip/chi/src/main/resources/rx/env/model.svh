class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  uvm_analysis_port #(flit_item) expected_ap, actual_ap;
  uvm_analysis_port #(bit) reset_ap;
  int granted = 0, occupied = 0;
  covergroup behavior with function sample (int count, bit stalled, bit simultaneous);
    option.per_instance = 1;
    occupancy: coverpoint count {
      bins empty = {0}; bins partial = {[1 : DEPTH - 1]}; bins full = {DEPTH};
    }
    sink_stall: coverpoint stalled;
    enqueue_and_dequeue: coverpoint simultaneous;
  endgroup
  function new(string name, uvm_component parent);
    super.new(name, parent);
    expected_ap = new("expected_ap", this);
    actual_ap = new("actual_ap", this);
    reset_ap = new("reset_ap", this);
    behavior = new();
  endfunction
  function void write(cycle_item t);
    link_signals s = t.sampled;
    flit_item item;
    bit retire = s.output_valid && s.output_ready;
    if (t.reset) begin
      reset_ap.write(1);
      granted  = 0;
      occupied = 0;
      return;
    end
    evidence::check(s.output_valid === (occupied != 0),
                    "CHI RX valid disagrees with buffered flits");
    if (s.input_valid) evidence::check(granted > 0, "CHI RX consumed an ungranted credit");
    behavior.sample(occupied, s.output_valid && !s.output_ready, s.input_valid && retire);
    if (s.input_valid) begin
      item = flit_item::type_id::create("expected");
      item.data = s.input_data;
      expected_ap.write(item);
    end
    if (retire) begin
      item = flit_item::type_id::create("actual");
      item.data = s.output_data;
      actual_ap.write(item);
    end
    granted += int'(s.credit) - int'(s.input_valid);
    occupied += int'(s.input_valid) - int'(retire);
    evidence::check(granted >= 0 && occupied >= 0 && granted + occupied <= DEPTH,
                    "CHI RX credit/storage conservation failed");
  endfunction
endclass
