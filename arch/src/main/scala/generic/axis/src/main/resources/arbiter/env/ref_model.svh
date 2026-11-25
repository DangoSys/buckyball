class ref_model extends reference_model #(cycle_item, result_item);
  `uvm_component_utils(ref_model)
  chandle model;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    model = axis_arbiter_create();
  endfunction

  function void write(cycle_item t);
    int unsigned selected;
    result_item  expected;
    if (t.reset) begin
      axis_arbiter_destroy(model);
      model = axis_arbiter_create();
      return;
    end
    selected = axis_arbiter_step(model, t.valid, t.ready, t.last);
    expected = result_item::type_id::create("expected");
    expected.valid = selected != 32'hffffffff;
    expected.ready = expected.valid && t.ready ? 2'b01 << selected : 0;
    expected.data = expected.valid ? t.data[selected] : 0;
    expected.keep = expected.valid ? t.keep[selected] : 0;
    expected.last = expected.valid ? t.last[selected] : 0;
    expected_ap.write(expected);
  endfunction

  function void final_phase(uvm_phase phase);
    axis_arbiter_destroy(model);
    super.final_phase(phase);
  endfunction
endclass
