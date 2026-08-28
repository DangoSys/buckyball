class fp2int_rows_test extends fp2int_case_test;
  `uvm_component_utils(fp2int_rows_test)

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
