class coverage extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(coverage)
  bit port1_stalled;
  bit packet_open;
  bit reset_pending;

  covergroup behavior with function sample (
      cycle_item t, bit late_competitor, bit locked_bubble, bit reset_held
  );
    option.per_instance = 1;
    contenders: coverpoint t.valid iff (!t.reset) {bins masks[] = {[0 : 3]};}
    winner: coverpoint t.accepted iff (!t.reset && t.ready) {
      bins idle = {0}; bins port0 = {1}; bins port1 = {2};
    }
    output_state: coverpoint {t.output_valid, t.ready} iff (!t.reset) {bins states[] = {[0 : 3]};}
    late_arrival: coverpoint late_competitor {bins seen = {1};}
    owner_gap: coverpoint locked_bubble {bins seen = {1};}
    reset_cancellation: coverpoint reset_held {bins seen = {1};}
    last_stall: coverpoint (t.output_valid && t.output_last && !t.ready) iff (!t.reset) {
      bins seen = {1};
    }
    keep0: coverpoint t.keep[0] iff (!t.reset && t.accepted[0]) {bins masks[] = {[0 : 15]};}
  endgroup

  function new(string name, uvm_component parent);
    super.new(name, parent);
    behavior = new();
  endfunction

  function void write(cycle_item t);
    behavior.sample(t, port1_stalled && t.valid == 3 && !t.reset,
                    packet_open && t.valid != 0 && !t.output_valid && !t.reset,
                    t.reset && reset_pending);
    if (t.reset) begin
      port1_stalled = 0;
      packet_open   = 0;
      reset_pending = 0;
    end else begin
      port1_stalled = t.valid == 2 && t.output_valid && !t.ready;
      if (t.output_valid) packet_open = !(t.ready && t.output_last);
      reset_pending = packet_open;
    end
  endfunction
endclass
