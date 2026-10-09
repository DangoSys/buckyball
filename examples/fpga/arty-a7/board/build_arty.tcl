# Run from project root: vivado -mode batch -source board/build_arty.tcl
set root [file normalize [file join [file dirname [info script]] ..]]
cd $root
set out [file join $root build arty]
file mkdir $out
set_param general.maxThreads 8
create_project -in_memory -part xc7a100tcsg324-1
set_property target_language Verilog [current_project]
read_verilog -sv [file join $root generated PebbleA7Chip.sv]
foreach f [glob [file join $root board *.sv]] { read_verilog -sv $f }
set_property verilog_define {SYNTHESIS BUCKYBALL_DISABLE_DPI} [current_fileset]
read_xdc [file join $root board Arty-A7-100T.xdc]
synth_design -top pebble_a7_top -part xc7a100tcsg324-1 -flatten_hierarchy rebuilt
write_checkpoint -force [file join $out synthesized.dcp]
report_utilization -hierarchical -file [file join $out utilization_synth.rpt]
report_timing_summary -file [file join $out timing_synth.rpt]
set unresolved [get_cells -hier -quiet -filter {IS_BLACKBOX == 1}]
if {[llength $unresolved]} { error "Unresolved cells: $unresolved" }
opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force [file join $out routed.dcp]
report_utilization -hierarchical -file [file join $out utilization_routed.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -file [file join $out timing_routed.rpt]
report_route_status -file [file join $out route_status.rpt]
report_drc -file [file join $out drc.rpt]
report_cdc -file [file join $out cdc.rpt]
report_io -file [file join $out io.rpt]
check_timing -verbose -file [file join $out check_timing.rpt]
foreach kind {max min} {
  set paths [get_timing_paths -delay_type $kind -max_paths 1]
  if {[llength $paths] == 0} { error "Missing $kind timing paths" }
  if {[get_property SLACK [lindex $paths 0]] < 0} { error "$kind timing failed; inspect timing_routed.rpt" }
}
set errors [get_drc_violations -quiet -filter {SEVERITY == Error}]
if {[llength $errors]} { error "DRC errors: $errors" }
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
write_bitstream -force [file join $out pebble_a7.bit]
puts "PASS: Arty A7 full-chip routed bitstream generated"
