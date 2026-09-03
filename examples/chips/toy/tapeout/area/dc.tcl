# Chip-owned DC flow. Timing comes from constraints.sdc (RUN_SDC).
# bbdev injects RUN_* via -x {set RUN_CONFIG <run.tcl>}.
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
set sram_cells [get_cells -hierarchical -quiet -filter {ref_name =~ "chipyard_sram_*"}]
if {[sizeof_collection $sram_cells] == 0} {
  error "no chipyard_sram_* cells after link; sram_leaves.v / link_library missing"
}
set_dont_touch $sram_cells
set sram_designs [get_designs -quiet chipyard_sram_*]
if {[sizeof_collection $sram_designs] > 0} {
  set_dont_touch $sram_designs
}
read_sdc [file normalize $RUN_SDC]
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
