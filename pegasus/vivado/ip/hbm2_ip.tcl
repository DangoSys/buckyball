# HBM2 IP configuration for AU280 (xcu280)
# Uses Stack 0, Pseudo-channel 0 only (MVP)
# AXI4 interface: 256-bit data, 33-bit address, 250 MHz
# Compatible with Vivado 2021.1 (HBM IP v1.0)

create_ip -name hbm -vendor xilinx.com -library ip -version 1.0 \
  -module_name hbm_0

set_property -dict [list \
  CONFIG.USER_HBM_STACK           {1} \
  CONFIG.USER_HBM_DENSITY         {8GB} \
  CONFIG.USER_SINGLE_STACK_SELECTION {LEFT} \
  CONFIG.USER_SWITCH_ENABLE_00    {TRUE} \
  CONFIG.USER_SWITCH_ENABLE_01    {FALSE} \
  CONFIG.USER_MC_ENABLE_00        {TRUE} \
  CONFIG.USER_MC_ENABLE_01        {FALSE} \
  CONFIG.USER_MC_ENABLE_02        {FALSE} \
  CONFIG.USER_MC_ENABLE_03        {FALSE} \
  CONFIG.USER_MC_ENABLE_04        {FALSE} \
  CONFIG.USER_MC_ENABLE_05        {FALSE} \
  CONFIG.USER_MC_ENABLE_06        {FALSE} \
  CONFIG.USER_MC_ENABLE_07        {FALSE} \
  CONFIG.USER_AXI_CLK_FREQ        {250} \
  CONFIG.USER_CLK_SEL_00          {FALSE} \
  CONFIG.USER_SAXI_00             {true} \
  CONFIG.USER_MC0_ENABLE_ECC_CORRECTION {TRUE} \
  CONFIG.USER_MC0_LOOKAHEAD_PCH   {TRUE} \
  CONFIG.USER_MC0_MAINTAIN_COHERENCY {TRUE} \
] [get_ips hbm_0]

generate_target all [get_ips hbm_0]
