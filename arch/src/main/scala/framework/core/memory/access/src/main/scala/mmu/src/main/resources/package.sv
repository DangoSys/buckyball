`include "parameters.svh"
`define MI(P, N) P[`MMU_IN_``N``_OFFSET+:`MMU_IN_``N``_WIDTH]
`define MO(P, N) P[`MMU_OUT_``N``_OFFSET+:`MMU_OUT_``N``_WIDTH]
`define RQ(N) `MMU_IN_REQ_BITS_``N``_OFFSET-`MMU_REQ_OFFSET+:`MMU_IN_REQ_BITS_``N``_WIDTH
package mmu_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  typedef bit [63:0] u64;
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
  `include "agent/item.svh"
  `include "agent/driver.svh"
  `include "agent/monitor.svh"
  `include "env/model.svh"
  `include "agent/peer.svh"
  `include "env/env.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
endpackage
