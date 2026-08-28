class int2fp_tensor_groups_test extends int2fp_case_test;
  `uvm_component_utils(int2fp_tensor_groups_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function int unsigned case_index();
    return 2;
  endfunction

  function string case_label();
    return "TENSOR_4_GROUPS";
  endfunction
endclass
