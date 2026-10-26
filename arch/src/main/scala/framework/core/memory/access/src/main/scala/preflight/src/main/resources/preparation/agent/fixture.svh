class fixture extends uvm_object;
  `uvm_object_utils(fixture)
  int policy[1<<`PREP_ID_BITS];
  bit pte_error;
  function new(string name = "fixture");
    super.new(name);
  endfunction
endclass
