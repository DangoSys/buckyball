class toint8_rows_test extends toint8_case_test;
  `uvm_component_utils(toint8_rows_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function int unsigned case_index();
    return 3;
  endfunction

  function string case_label();
    return "ROWS";
  endfunction
endclass
