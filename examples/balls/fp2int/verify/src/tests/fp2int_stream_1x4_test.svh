class fp2int_stream_1x4_test extends fp2int_case_test;
  `uvm_component_utils(fp2int_stream_1x4_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function int unsigned case_index();
    return 5;
  endfunction

  function string case_label();
    return "STREAM_1X4";
  endfunction
endclass
