class result_item extends uvm_sequence_item;
  bit [511:0] data;
  bit error;
  int tag, client;
  `uvm_object_utils_begin(result_item)

    `uvm_field_int(data, UVM_DEFAULT)
    `uvm_field_int(error, UVM_DEFAULT)

    `uvm_field_int(tag, UVM_DEFAULT)
    `uvm_field_int(client, UVM_DEFAULT)

  `uvm_object_utils_end
  function new(string name = "result_item");
    super.new(name);
  endfunction
endclass
