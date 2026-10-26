class ref_model extends reference_model #(operation, item);
  `uvm_component_utils(ref_model)
  chandle model;
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    model = bank_ref_create(`BANK_WORDS);
  endfunction
  function void write(operation t);
    int unsigned result = bank_ref_access(model, t.word_addr, t.write, t.data, t.mask);
    if (!t.write) begin
      item expected = item::type_id::create("expected");
      expected.data = result;
      expected.keep = 15;
      expected.last = t.last;
      expected_ap.write(expected);
    end
  endfunction
  function void final_phase(uvm_phase phase);
    bank_ref_destroy(model);
    super.final_phase(phase);
  endfunction
endclass
