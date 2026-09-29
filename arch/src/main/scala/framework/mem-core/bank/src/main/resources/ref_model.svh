class ref_model extends reference_model #(request_item, response_item);
  `uvm_component_utils(ref_model)
  chandle model;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    model = bank_ref_create(16);
  endfunction

  function void write(request_item t);
    response_item expected = response_item::type_id::create("expected");
    expected.data  = bank_ref_access(model, t.addr, t.write, t.data, t.mask);
    expected.tag   = t.tag;
    expected.error = 0;
    expected_ap.write(expected);
  endfunction

  function void final_phase(uvm_phase phase);
    bank_ref_destroy(model);
    super.final_phase(phase);
  endfunction
endclass
