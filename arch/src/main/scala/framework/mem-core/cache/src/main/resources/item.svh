class request_item extends uvm_sequence_item;
  logic [`CACHE_ID_BITS-1:0] id;
  logic [2:0] op;
  logic [`CACHE_ADDR_BITS-1:0] addr;
  logic [`CACHE_WAY_BITS-1:0] way;
  bit [`CACHE_LINE_BITS-1:0] data;
  bit [`CACHE_LINE_BYTES-1:0] mask;
  logic [`CACHE_META_BITS-1:0] metadata;
  bit [`CACHE_WAYS-1:0] eligible;
  `uvm_object_utils(request_item)
  function new(string name = "request_item");
    super.new(name);
  endfunction
endclass

class response_item extends uvm_sequence_item;
  logic [`CACHE_ID_BITS-1:0] id;
  logic hit, available, entry_valid;
  logic [`CACHE_WAY_BITS-1:0] way;
  logic [`CACHE_ADDR_BITS-1:0] addr;
  logic [`CACHE_LINE_BITS-1:0] data;
  logic [`CACHE_META_BITS-1:0] metadata;
  `uvm_object_utils_begin(response_item)
    `uvm_field_int(id, UVM_ALL_ON)
    `uvm_field_int(hit, UVM_ALL_ON)
    `uvm_field_int(available, UVM_ALL_ON)
    `uvm_field_int(entry_valid, UVM_ALL_ON)
    `uvm_field_int(way, UVM_ALL_ON)
    `uvm_field_int(addr, UVM_ALL_ON)
    `uvm_field_int(data, UVM_ALL_ON)
    `uvm_field_int(metadata, UVM_ALL_ON)
  `uvm_object_utils_end
  function new(string name = "response_item");
    super.new(name);
  endfunction
endclass
