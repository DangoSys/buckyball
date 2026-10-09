set root [file normalize [file join [file dirname [info script]] ..]]
cd $root
open_checkpoint build/arty/routed.dcp
if {[get_property PART [current_design]] ne "xc7a100tcsg324-1"} { error "Wrong part" }
set expected {CLK100MHZ E3 btn0 D9 sw0 A8 uart_txd_in A9 uart_rxd_out D10 {led[0]} H5 {led[1]} J5 {led[2]} T9 {led[3]} T10}
if {[llength [get_ports]] != 9} { error "Unexpected board ports" }
foreach {port pin} $expected {
  set actual [get_property PACKAGE_PIN [get_ports $port]]
  if {$actual ne $pin} { error "Wrong pin for $port: $actual instead of $pin" }
  if {[get_property IOSTANDARD [get_ports $port]] ne "LVCMOS33"} { error "Wrong IO voltage standard for $port" }
}
set clocks [get_clocks -filter {NAME == clk_out}]
if {[llength $clocks] != 1 || [get_property PERIOD $clocks] != 20.000} { error "Core clock is not 50 MHz" }
set blackboxes [get_cells -hier -quiet -filter {IS_BLACKBOX == 1}]
set latches [get_cells -hier -quiet -filter {REF_NAME =~ LD*}]
if {[llength $blackboxes] || [llength $latches]} { error "Unexpected blackbox/latch" }
report_utilization -file build/arty/utilization_flat.rpt
set file [open build/arty/audit.txt w]
puts $file "PASS: xc7a100tcsg324-1; all 9 board pins and LVCMOS33 verified; clk_out=20ns; no blackboxes/latches"
close $file
close_design
