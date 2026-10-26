class command extends uvm_sequence_item;
  `uvm_object_utils(command)
  bit write;
  int unsigned word_addr;
  int unsigned beats;
  int unsigned completion_stall;
  function new(string name = "command");
    super.new(name);
  endfunction
endclass
