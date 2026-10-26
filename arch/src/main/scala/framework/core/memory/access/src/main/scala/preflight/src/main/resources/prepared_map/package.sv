`include "parameters.svh"
`define PF(K, F) `PF_``K``_``F``_OFFSET +: `PF_``K``_``F``_WIDTH
package prepared_map_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  `include "uvm_macros.svh"
  typedef longint unsigned u64;
  import "DPI-C" function chandle pmap_ref_create(input int unsigned address_bits);
  import "DPI-C" function void pmap_ref_destroy(input chandle model);
  import "DPI-C" function void pmap_ref_reset(input chandle model);
  import "DPI-C" function void pmap_ref_reserve(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function void pmap_ref_prepared(
    input chandle model,
    input int unsigned id,
    input u64 va,
    pa,
    input int unsigned bytes,
    write,
    last,
    error
  );
  import "DPI-C" function int unsigned pmap_ref_ready(
    input chandle model,
    input int unsigned id,
    fire,
    output int unsigned error,
    output u64 va
  );
  import "DPI-C" function void pmap_ref_query(
    input chandle model,
    input int unsigned valid,
    id,
    input u64 va,
    input int unsigned bytes,
    write,
    retiring,
    retiring_id,
    output int unsigned hit,
    output u64 pa,
    output int unsigned error
  );
  import "DPI-C" function void pmap_ref_release(
    input chandle model,
    input int unsigned id
  );
  import "DPI-C" function int unsigned pmap_ref_pending(input chandle model);
  `include "env/env.svh"
  `include "tests/test.svh"
endpackage
