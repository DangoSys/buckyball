class int2fp_tensor_rows_test extends int2fp_case_test;
  `uvm_component_utils(int2fp_tensor_rows_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function int unsigned case_index();
    return 0;
  endfunction

  function string case_label();
    return "TENSOR_16_ROWS";
  endfunction
endclass
