package storage_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  typedef virtual storage_if storage_vif;

  class transaction extends uvm_sequence_item;
    bit [127:0] data;
    bit error;
    `uvm_object_utils_begin(transaction)
      `uvm_field_int(data, UVM_ALL_ON)
      `uvm_field_int(error, UVM_ALL_ON)
    `uvm_object_utils_end
    function new(string name = "transaction");
      super.new(name);
    endfunction
  endclass

  `include "storage_test.svh"
endpackage
