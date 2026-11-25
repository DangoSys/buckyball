class fixture extends uvm_object;
  `uvm_object_utils(fixture)
  bit error[`TR_CLIENTS+`TR_EXTERNAL];
  bit move_error;
  function new(string name = "fixture");
    super.new(name);
  endfunction
endclass
