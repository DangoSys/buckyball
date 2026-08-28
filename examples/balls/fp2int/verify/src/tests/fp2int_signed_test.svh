class fp2int_signed_test extends fp2int_case_test;
  `uvm_component_utils(fp2int_signed_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function int unsigned case_index();
    return 0;
  endfunction

  function string case_label();
    return "SIGNED";
  endfunction
endclass
