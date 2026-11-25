class operation extends uvm_sequence_item;
  bit write;
  int unsigned word_addr;
  int unsigned data;
  byte unsigned mask;
  bit last;
  `uvm_object_utils(operation)
  function new(string name = "operation");
    super.new(name);
  endfunction
endclass
