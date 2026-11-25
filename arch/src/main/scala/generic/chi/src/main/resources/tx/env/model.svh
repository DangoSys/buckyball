class model extends uvm_subscriber #(cycle_item);
  `uvm_component_utils(model)
  uvm_analysis_port #(flit_item) expected_ap, actual_ap;
  uvm_analysis_port #(bit) reset_ap;
  int balance = 0;
  bit delayed_credit = 0, delayed_send = 0;
  covergroup behavior with function sample (int credits, bit stalled, bit simultaneous);
    option.per_instance = 1;
    available: coverpoint credits {
      bins empty = {0}; bins partial = {[1 : DEPTH - 1]}; bins full = {DEPTH};
    }
    source_stall: coverpoint stalled;
    credit_and_send: coverpoint simultaneous;
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
    bit fire = s.input_valid && s.input_ready;
    if (t.reset) begin
      reset_ap.write(1);
      balance = 0;
      delayed_credit = 0;
      delayed_send = 0;
      return;
    end
    evidence::check(s.credits == balance, "CHI TX credit count mismatch");
    evidence::check(s.input_ready === (s.active && balance != 0),
                    "CHI TX ready disagrees with available credit");
    evidence::check(s.pend === s.active, "CHI TX FLITPEND disagrees with activation");
    evidence::check(s.output_valid === delayed_send,
                    "CHI TX FLITV did not follow the accepted flit");
    behavior.sample(balance, s.input_valid && !s.input_ready, delayed_credit && fire);
    if (fire) begin
      item = flit_item::type_id::create("expected");
      item.data = s.input_data;
      expected_ap.write(item);
    end
    if (s.output_valid) begin
      item = flit_item::type_id::create("actual");
      item.data = s.output_data;
      actual_ap.write(item);
    end
    balance += int'(delayed_credit) - int'(fire);
    evidence::check(balance >= 0 && balance <= DEPTH, "CHI TX credit conservation failed");
    delayed_credit = s.credit;
    delayed_send   = fire;
  endfunction
endclass
