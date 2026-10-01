`include "coherence_config.svh"
`define CF(K, F) `COH_``K``_``F``_OFFSET +: `COH_``K``_``F``_WIDTH
package coherence_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  typedef logic [`COH_REQ_WIDTH-1:0] req_t;
  typedef logic [`COH_RSP_WIDTH-1:0] rsp_t;
  typedef logic [`COH_DAT_WIDTH-1:0] dat_t;
  typedef logic [`COH_MEMRESP_WIDTH-1:0] mem_t;
  typedef struct packed {
    bit valid, unique_owner, dirty;
    bit [511:0] data;
  } private_line_t;
  import "DPI-C" function chandle coherence_ref_create();
  import "DPI-C" function void coherence_ref_destroy(input chandle model);
  import "DPI-C" function void coherence_ref_initial(
    input longint unsigned addr,
    output bit [511:0] data
  );
  import "DPI-C" function void coherence_ref_read(
    input chandle model,
    input longint unsigned addr,
    output bit [511:0] data
  );
  import "DPI-C" function void coherence_ref_write(
    input chandle model,
    input longint unsigned addr,
    input bit [511:0] data
  );

  class completion extends uvm_sequence_item;
    int node, txn, opcode;
    bit is_data;
    logic [2:0] permission = 0;
    logic [1:0] error = 0;
    logic [511:0] data = 0;
    `uvm_object_utils_begin(completion)
      `uvm_field_int(node, UVM_ALL_ON)
      `uvm_field_int(txn, UVM_ALL_ON)
      `uvm_field_int(opcode, UVM_ALL_ON)
      `uvm_field_int(is_data, UVM_ALL_ON)
      `uvm_field_int(permission, UVM_ALL_ON)
      `uvm_field_int(error, UVM_ALL_ON)
      `uvm_field_int(data, UVM_ALL_ON)
    `uvm_object_utils_end
    function new(string name = "completion");
      super.new(name);
    endfunction
  endclass

  `include "env.svh"
  `include "test.svh"
endpackage
