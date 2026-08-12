# Shared Design Compiler implementation. Chip tapeout/dc.tcl files are thin
# ownership wrappers; all RUN_* values are supplied by bbdev.
if {![info exists RUN_CONFIG] || $RUN_CONFIG eq ""} {
  error "bbdev must pass -x {set RUN_CONFIG <run.tcl>}"
}
source [file normalize $RUN_CONFIG]

proc bbdev_read_filelist {path} {
  set fh [open $path r]
  set files [list]
  while {[gets $fh line] >= 0} {
    set line [string trim $line]
    if {$line eq "" || [string match "#*" $line]} { continue }
    lappend files [file normalize $line]
  }
  close $fh
  if {[llength $files] == 0} { error "empty DC source list: $path" }
  return $files
}

set top $RUN_TOP
set output_dir [file normalize $RUN_OUTPUT_DIR]
set report_dir [file normalize $RUN_REPORT_DIR]
file mkdir $output_dir $report_dir [file join $report_dir work]
set target_library [list $RUN_TARGET_LIBRARY]
set synthetic_library $RUN_SYNTHETIC_LIBRARY
set link_library [concat [list *] $target_library $synthetic_library $RUN_LINK_LIBRARY]
set_host_options -max_cores $RUN_MAX_CORES
define_design_lib WORK -path [file join $report_dir work]
set search_path [list .]
set_app_var verilogout_no_tri true
set_app_var verilogout_equation false
analyze -format sverilog -define {SYNTHESIS DC_SYN} [bbdev_read_filelist $RUN_SOURCE_LIST]
elaborate $top
current_design $top
link
set bb_clock [get_ports -quiet $RUN_CLOCK_PORT]
if {[sizeof_collection $bb_clock] != 1} { error "clock port '$RUN_CLOCK_PORT' was not found exactly once" }
create_clock -name bb_clock -period $RUN_CLOCK_PERIOD_NS $bb_clock
set_clock_uncertainty [expr {$RUN_CLOCK_PERIOD_NS * 0.30}] [get_clocks bb_clock]
set_clock_transition [expr {$RUN_CLOCK_PERIOD_NS * 0.10}] [get_clocks bb_clock]
set_input_delay [expr {$RUN_CLOCK_PERIOD_NS * 0.70}] -clock bb_clock [remove_from_collection [all_inputs] $bb_clock]
set_output_delay [expr {$RUN_CLOCK_PERIOD_NS * 0.70}] -clock bb_clock [all_outputs]
set_load 2.0 [all_outputs]
compile_ultra -area_high_effort_script -no_autoungroup -no_boundary_optimization
set_fix_multiple_port_nets -all -buffer_constants
change_names -hierarchy -rules verilog
write -format ddc -hierarchy -output [file join $output_dir ${top}.ddc]
write -format verilog -hierarchy -output [file join $output_dir ${top}.v]
write_sdc [file join $output_dir ${top}.sdc]
report_constraint -all_violators > [file join $report_dir constraint.rpt]
report_timing -delay max -max_paths 50 > [file join $report_dir timing_max.rpt]
report_timing -delay min -max_paths 50 > [file join $report_dir timing_min.rpt]
report_area -hierarchy > [file join $report_dir area.rpt]
report_reference > [file join $report_dir reference.rpt]
exit
