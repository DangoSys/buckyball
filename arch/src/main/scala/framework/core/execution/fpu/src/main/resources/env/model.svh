class model extends reference_model #(request_item, response_item);
  `uvm_component_utils(model)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction
  function void write(request_item req);
    response_item rsp = response_item::type_id::create("expected");
    longint unsigned value;
    int unsigned flags;
    byte unsigned integer_result, illegal;
    fpu_expected(req.instruction, req.integer_value, req.a, req.rhs, req.c, req.rounding, value,
                 flags, integer_result, illegal);
    rsp.value = illegal ? 0 : value;
    rsp.flags = flags[4:0];
    rsp.integer_result = illegal ? 0 : integer_result[0];
    rsp.illegal = illegal[0];
    expected_ap.write(rsp);
  endfunction
endclass
