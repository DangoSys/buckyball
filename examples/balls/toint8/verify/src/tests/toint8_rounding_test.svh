class toint8_rounding_test extends toint8_case_test;
  `uvm_component_utils(toint8_rounding_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function int unsigned case_index();
    return 2;
  endfunction

  function string case_label();
    return "ROUNDING";
  endfunction
endclass
