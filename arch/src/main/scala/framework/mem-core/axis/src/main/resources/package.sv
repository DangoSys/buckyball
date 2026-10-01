package axis_pkg;
  import uvm_pkg::*;
  import ip_pkg::*;
  import axis_vip_pkg::*;
  typedef axis_item#(32) item;
  typedef axis_source_agent#(32) source_agent;
  typedef axis_monitor#(32) monitor;
  typedef axis_sink#(32) sink;
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

  `include "ref_model.svh"
  `include "env.svh"
  `include "test.svh"
endpackage
