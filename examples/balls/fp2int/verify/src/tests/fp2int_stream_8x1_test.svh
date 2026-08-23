class fp2int_stream_8x1_test extends fp2int_case_test;
  `uvm_component_utils(fp2int_stream_8x1_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function int unsigned case_index();
    return 7;
  endfunction

  function string case_label();
    return "STREAM_8X1";
  endfunction
endclass
