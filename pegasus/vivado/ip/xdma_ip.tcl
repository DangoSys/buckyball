# XDMA PCIe IP configuration for AU280
# PCIe x16 Gen3, AXI4 data width 512-bit, 250 MHz AXI clock
# AXI-Lite master (BAR0) + AXI4 DMA master + AXI-Stream C2H channel
# Compatible with Vivado 2021.1 (XDMA IP v4.1)

create_ip -name xdma -vendor xilinx.com -library ip -version 4.1 \
  -module_name xdma_0

set_property -dict [list \
  CONFIG.pl_link_cap_max_link_width {X16} \
  CONFIG.pl_link_cap_max_link_speed {8.0_GT/s} \
  CONFIG.axi_data_width           {512_bit} \
  CONFIG.axisten_freq             {250} \
  CONFIG.axilite_master_en        {true} \
  CONFIG.axilite_master_size      {32} \
  CONFIG.xdma_axi_intf_mm         {AXI_Memory_Mapped} \
  CONFIG.xdma_rnum_chnl           {1} \
  CONFIG.xdma_wnum_chnl           {1} \
  CONFIG.pf0_msix_cap_table_bir   {BAR_1} \
  CONFIG.pf0_msix_cap_pba_bir     {BAR_1} \
  CONFIG.dsc_bypass_rd            {0000} \
  CONFIG.dsc_bypass_wr            {0000} \
] [get_ips xdma_0]

generate_target all [get_ips xdma_0]
