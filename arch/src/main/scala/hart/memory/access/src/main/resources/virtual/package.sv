`include "virtual_parameters.svh"
`define VI(F) vif.inputs[`VA_IN_``F``_OFFSET+:`VA_IN_``F``_WIDTH]
`define VO(F) vif.outputs[`VA_OUT_``F``_OFFSET+:`VA_OUT_``F``_WIDTH]
`define VB(F) vif.backend[`VA_IN_``F``_OFFSET+:`VA_IN_``F``_WIDTH]
`define VS(F) pins[`VA_IN_``F``_OFFSET+:`VA_IN_``F``_WIDTH]
`define MI(T, F) T.inputs[`VA_IN_``F``_OFFSET+:`VA_IN_``F``_WIDTH]
`define MO(T, F) T.outputs[`VA_OUT_``F``_OFFSET+:`VA_OUT_``F``_WIDTH]
package virtual_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  import "DPI-C" function chandle mmu_ref_create(
    input int unsigned address_bits,
    entries
  );
  import "DPI-C" function void mmu_ref_destroy(input chandle model);
  import "DPI-C" function void mmu_ref_clear(input chandle model);
  import "DPI-C" function void mmu_ref_program(
    input chandle model,
    input longint unsigned address,
    value,
    input int unsigned error
  );
  import "DPI-C" function int unsigned mmu_ref_read(
    input chandle model,
    input longint unsigned address,
    output longint unsigned value,
    output int unsigned error
  );
  import "DPI-C" function int unsigned mmu_ref_pending(input chandle model);
  import "DPI-C" function int unsigned mmu_ref_peek(
    input chandle model,
    output longint unsigned address
  );
  import "DPI-C" function int unsigned mmu_ref_consume(
    input chandle model,
    input longint unsigned address
  );
  import "DPI-C" function int unsigned mmu_ref_translate(
    input chandle model,
    input longint unsigned address,
    input int unsigned mode,
    input longint unsigned root,
    input int unsigned privilege,
    wr,
    execute,
    sum,
    mxr,
    output longint unsigned paddr,
    output int unsigned pf,
    af,
    level
  );
  import "DPI-C" function int unsigned mmu_ref_lookup(
    input chandle model,
    input longint unsigned address,
    input int unsigned mode,
    input longint unsigned root,
    input int unsigned privilege,
    wr,
    execute,
    sum,
    mxr,
    output longint unsigned paddr
  );
  import "DPI-C" function int unsigned mmu_ref_cached(input chandle model);
  import "DPI-C" function void mmu_ref_result(input chandle model);
  import "DPI-C" function void mmu_ref_reset(input chandle model);
  import "DPI-C" function void mmu_ref_flush(input chandle model);
  import "DPI-C" function void mem_access_ref_prepare(
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
  import "DPI-C" function longint unsigned mem_access_ref_result(
    input longint unsigned addr,
    input int unsigned size,
    write,
    signed_load,
    atomic_op,
    cacheable,
    input longint unsigned raw,
    input int unsigned error
  );
  `include "agent/item.svh"
  `include "agent/fixture.svh"
  `include "agent/driver.svh"
  `include "agent/monitor.svh"
  `include "agent/peer.svh"
  `include "env/model.svh"
  `include "env/env.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
endpackage
