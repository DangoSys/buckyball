`include "cache_config.svh"
package cache_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  `include "uvm_macros.svh"
  localparam LOOKUP = 0, READ = 1, WRITE = 2, FILL = 3, INVALIDATE = 4;
  import "DPI-C" function chandle cache_ref_create(
    input int unsigned sets,
    ways,
    line_bytes
  );
  import "DPI-C" function void cache_ref_destroy(input chandle model);
  import "DPI-C" function void cache_ref_reset(input chandle model);
  import "DPI-C" function void cache_ref_access(
    input chandle model,
    input int unsigned op,
    input longint unsigned addr,
    input int unsigned way,
    input bit [`CACHE_LINE_BITS-1:0] data,
    input bit [`CACHE_LINE_BYTES-1:0] mask,
    input int unsigned metadata,
    input bit [`CACHE_WAYS-1:0] eligible,
    output byte unsigned hit,
    available,
    output int unsigned out_way,
    output byte unsigned entry_valid,
    output longint unsigned out_addr,
    output bit [`CACHE_LINE_BITS-1:0] out_data,
    output int unsigned out_metadata
  );
  `include "item.svh"
  `include "ref_model.svh"
  `include "env.svh"
  `include "test.svh"
endpackage
