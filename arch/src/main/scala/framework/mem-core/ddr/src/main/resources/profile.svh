`ifndef DDR_PROFILE_SVH
`define DDR_PROFILE_SVH
`ifdef DDR_PROFILE64
`define DDR_CONFIG "ddr64_config.svh"
`define DDR_PORTS "ddr64_ports.svh"
`elsif DDR_PROFILE256
`define DDR_CONFIG "ddr256_config.svh"
`define DDR_PORTS "ddr256_ports.svh"
`elsif DDR_PROFILE512
`define DDR_CONFIG "ddr512_config.svh"
`define DDR_PORTS "ddr512_ports.svh"
`elsif DDR_PROFILESIX
`define DDR_CONFIG "ddr_six_config.svh"
`define DDR_PORTS "ddr_six_ports.svh"
`else
`define DDR_CONFIG "ddr_config.svh"
`define DDR_PORTS "ddr_ports.svh"
`endif
`include `DDR_CONFIG
`endif
