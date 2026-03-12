# Xilinx AU280 (xcu280-fsvh2892-2L-e) Pin and Timing Constraints
# For Pegasus FPGA simulation framework

################################################################################
# PCIe Interface (x16, connected to AU280 PCIe edge connector)
################################################################################

# PCIe reference clock (100 MHz differential, from PCIe slot)
set_property PACKAGE_PIN BF41 [get_ports pcie_sys_clk_p]
set_property PACKAGE_PIN BG41 [get_ports pcie_sys_clk_n]
create_clock -period 10.000 -name pcie_refclk [get_ports pcie_sys_clk_p]

# PCIe reset (active low, from PCIe slot)
set_property PACKAGE_PIN BJ44  [get_ports pcie_sys_rst_n]
set_property IOSTANDARD LVCMOS12 [get_ports pcie_sys_rst_n]
set_property PULLUP true [get_ports pcie_sys_rst_n]
set_false_path -from [get_ports pcie_sys_rst_n]

# PCIe TX/RX differential pairs (x16)
# Note: actual pin assignments depend on AU280 board schematic / Vivado auto-assign
# These are placeholders; real assignments come from xdma IP's XDC output
# set_property PACKAGE_PIN <pin> [get_ports {pcie_exp_txp[0]}]
# ... (x16 lanes, auto-assigned by XDMA IP)

################################################################################
# HBM2 Reference Clock (from MMCM or direct board clock)
################################################################################

# HBM2 cattrip override (required for AU280 to prevent thermal shutdown)
# Must be driven by MMCM output at 100 MHz
# set_property PACKAGE_PIN <pin> [get_ports hbm_cattrip]

################################################################################
# Main AXI Clock (250 MHz from XDMA IP axi_aclk output)
################################################################################

# The axi_aclk is generated inside XDMA IP — create a generated clock constraint
create_generated_clock -name axi_aclk \
  -source [get_pins xdma_0/inst/pcie4_ip_i/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O] \
  -multiply_by 1 \
  [get_pins xdma_0/inst/pcie4_ip_i/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]

################################################################################
# DUT Clock (gated by SCU BUFGCE, derived from axi_aclk)
################################################################################

# The DUT clock is axi_aclk gated by BUFGCE.
# Vivado will automatically detect the BUFGCE hierarchy and propagate the clock.
# Add a multicycle path constraint since BUFGCE introduces enable-based gating.
set_multicycle_path -setup 1 \
  -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */bufgce/O}]] \
  -to   [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */bufgce/O}]]

################################################################################
# False paths (async reset signals)
################################################################################

set_false_path -from [get_ports pcie_sys_rst_n]

################################################################################
# I/O Timing (set_input_delay / set_output_delay not needed for PCIe — XDMA IP handles it)
################################################################################

################################################################################
# Placement hints
################################################################################

# Place XDMA IP near PCIe lanes (right side of AU280)
# Place HBM2 controller near HBM2 stacks (left side of AU280)
# These are Pblock suggestions; actual placement by Vivado

################################################################################
# Bitstream settings
################################################################################

set_property BITSTREAM.CONFIG.SPI_BUSWIDTH  4    [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE    33   [current_design]
set_property CONFIG_VOLTAGE                 1.8  [current_design]
set_property CFGBVS                         GND  [current_design]
