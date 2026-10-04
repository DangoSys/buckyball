`include "cpu_mem_config.svh"
package cpu_mem_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  `define CPUF(kind, field) `CPU_``kind``_``field``_OFFSET +: `CPU_``kind``_``field``_WIDTH
  typedef bit [`CPU_REQ_WIDTH-1:0] req_t;
  import "DPI-C" function void cpu_mem_ref_prepare(
    input longint unsigned addr,
    input int unsigned size,
    write,
    input longint unsigned data,
    input int unsigned atomic_op,
    cacheable,
    normal,
    address_bits,
    output int unsigned target,
    misaligned,
    access_fault,
    output longint unsigned bus_addr,
    bus_data,
    output int unsigned bus_mask,
    atomic_word
  );
  import "DPI-C" function longint unsigned cpu_mem_ref_result(
    input longint unsigned addr,
    input int unsigned size,
    write,
    signed_load,
    atomic_op,
    cacheable,
    input longint unsigned raw,
    input int unsigned error
  );
  `include "test.svh"
endpackage
