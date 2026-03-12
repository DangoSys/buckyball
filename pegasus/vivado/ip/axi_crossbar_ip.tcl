# AXI Crossbar IP configuration
# 2 master ports (XDMA DMA + ChipTop ExtMem), 1 slave port (HBM2 PC0)
# Data width: 256-bit, address width: 33-bit, ID width: 6-bit, 250 MHz

create_ip -name axi_crossbar -vendor xilinx.com -library ip -version 2.1 \
  -module_name axi_crossbar_0

set_property -dict [list \
  CONFIG.NUM_SI               {2} \
  CONFIG.NUM_MI               {1} \
  CONFIG.STRATEGY             {1} \
  CONFIG.DATA_WIDTH           {256} \
  CONFIG.ADDR_WIDTH           {33} \
  CONFIG.ID_WIDTH             {6} \
  CONFIG.M00_A00_BASE_ADDR    {0x0000000000000000} \
  CONFIG.M00_A00_ADDR_WIDTH   {33} \
  CONFIG.AWUSER_WIDTH         {0} \
  CONFIG.ARUSER_WIDTH         {0} \
  CONFIG.WUSER_WIDTH          {0} \
  CONFIG.RUSER_WIDTH          {0} \
  CONFIG.BUSER_WIDTH          {0} \
] [get_ips axi_crossbar_0]

generate_target all [get_ips axi_crossbar_0]

# AXI Data Width Converter: 512-bit (XDMA) → 256-bit (HBM2/Crossbar)
create_ip -name axi_dwidth_converter -vendor xilinx.com -library ip -version 2.1 \
  -module_name axi_dwidth_converter_0

set_property -dict [list \
  CONFIG.ADDR_WIDTH          {33} \
  CONFIG.MI_DATA_WIDTH       {256} \
  CONFIG.SI_DATA_WIDTH       {512} \
  CONFIG.SI_ID_WIDTH         {4} \
] [get_ips axi_dwidth_converter_0]

generate_target all [get_ips axi_dwidth_converter_0]
