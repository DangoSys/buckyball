`ifndef GEMMINI_CONFIG_FUNCT7
`error "GEMMINI_CONFIG_FUNCT7 must be provided by the selected Thread ballISA"
`endif
`ifndef GEMMINI_LOOP_WS_CONFIG_BOUNDS_FUNCT7
`error "GEMMINI_LOOP_WS_CONFIG_BOUNDS_FUNCT7 must be provided by the selected Thread ballISA"
`endif
`ifndef GEMMINI_PRELOAD_FUNCT7
`error "GEMMINI_PRELOAD_FUNCT7 must be provided by the selected Thread ballISA"
`endif
`ifndef GEMMINI_COMPUTE_PRELOADED_FUNCT7
`error "GEMMINI_COMPUTE_PRELOADED_FUNCT7 must be provided by the selected Thread ballISA"
`endif

// The cycle model of BEMU (emu/src/latency.rs), through the casegen library.
import "DPI-C" function longint unsigned gemmini_bemu_cycles(
    input longint unsigned iter, input longint unsigned rs2, input byte unsigned ws,
    input byte unsigned a_transpose, input byte unsigned b_transpose);
import "DPI-C" function longint unsigned gemmini_bemu_loop_config_cycles();
import "DPI-C" function longint unsigned gemmini_bemu_loop_ws_cycles(
    input longint unsigned max_i, input longint unsigned max_j, input longint unsigned max_k);
import "DPI-C" function longint unsigned gemmini_bemu_loop_conv_ws_cycles(
    input longint unsigned batch, input longint unsigned out_dim,
    input longint unsigned out_channels, input longint unsigned in_channels,
    input longint unsigned kernel_dim);
