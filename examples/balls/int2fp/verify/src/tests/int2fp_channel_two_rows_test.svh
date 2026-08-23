class int2fp_channel_two_rows_test extends int2fp_case_test;
  `uvm_component_utils(int2fp_channel_two_rows_test)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function int unsigned case_index();
    return 4;
  endfunction

  function string case_label();
    return "CHANNEL_TWO_ROWS";
  endfunction
endclass
