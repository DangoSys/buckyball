package axis_pkg;
  import uvm_pkg::*;
  import verification_pkg::*;
  import axis_vip_pkg::*;
  typedef axis_item#(`AXIS_DATA_BITS) item;
  typedef axis_source_agent#(`AXIS_DATA_BITS) source_agent;
  typedef axis_monitor#(`AXIS_DATA_BITS) monitor;
  typedef axis_sink#(`AXIS_DATA_BITS) sink;
  `include "uvm_macros.svh"

  import "DPI-C" function chandle axis_ref_create();
  import "DPI-C" function void axis_ref_destroy(input chandle model);
  import "DPI-C" function void axis_ref_push(
    input chandle model,
    input int unsigned data,
    input byte unsigned keep,
    input byte unsigned last
  );
  import "DPI-C" function byte unsigned axis_ref_pop(
    input chandle model,
    output int unsigned data,
    output byte unsigned keep,
    output byte unsigned last
  );

  `include "env/ref_model.svh"
  `include "../common/coverage.svh"
  `include "env/env.svh"
  `include "seq/sequence.svh"
  `include "tests/test.svh"
endpackage
