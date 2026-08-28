class int2fp_channel_base_test extends int2fp_case_test;
  `uvm_component_utils(int2fp_channel_base_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function int unsigned case_index();
    return 3;
  endfunction

  function string case_label();
    return "CHANNEL_BASE_64";
  endfunction
endclass
