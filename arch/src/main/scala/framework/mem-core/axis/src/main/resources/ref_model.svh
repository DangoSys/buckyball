class ref_model extends uvm_subscriber #(item);
  `uvm_component_utils(ref_model)

  chandle model;
  uvm_analysis_port #(item) expected_ap;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    expected_ap = new("expected_ap", this);
    model = axis_ref_create();
    if (model == null) begin
      `uvm_fatal("AXIS_MODEL", "failed to create Rust reference model")
    end
  endfunction

  function void write(item t);
    item expected;
    int unsigned data;
    byte unsigned keep;
    byte unsigned last;

    axis_ref_push(model, t.data, t.keep, t.last);
    if (!axis_ref_pop(model, data, keep, last)) begin
      `uvm_fatal("AXIS_MODEL", "reference model produced no output")
    end
    expected = item::type_id::create("expected");
    expected.data = data;
    expected.keep = keep[3:0];
    expected.last = last[0];
    expected_ap.write(expected);
  endfunction

  function void final_phase(uvm_phase phase);
    axis_ref_destroy(model);
    super.final_phase(phase);
  endfunction
endclass
