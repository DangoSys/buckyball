`ifndef MATADD_FUNCT7
`error "MATADD_FUNCT7 must be provided by the selected Thread ballISA"
`endif

localparam int MATADD_THREAD_FUNCT7 = `MATADD_FUNCT7;
localparam int MATADD_MAX_LINES = 64;
localparam int MATADD_TIMEOUT_CYCLES = 100000;
