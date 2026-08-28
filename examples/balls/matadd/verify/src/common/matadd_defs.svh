`ifndef MATADD_FUNCT7
`error "MATADD_FUNCT7 must be provided by the selected Core ballISA"
`endif

localparam int MATADD_CORE_FUNCT7 = `MATADD_FUNCT7;
localparam int MATADD_MAX_LINES = 1024;
localparam int MATADD_TIMEOUT_CYCLES = 100000;

function automatic int unsigned matadd_require_bid();
  int unsigned bid;
  if (!$value$plusargs("BID=%d", bid)) begin
    `uvm_fatal("BID", "missing required plusarg +BID=<n>")
  end
  return bid;
endfunction
