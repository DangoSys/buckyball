# Shared PrimeTime PX implementation. Design tapeout/power.tcl files are thin
# wrappers; this reports only standard-cell dynamic power.
if {![info exists RUN_CONFIG] || $RUN_CONFIG eq ""} { error "bbdev must pass -x {set RUN_CONFIG <power-run.tcl>}" }
source [file normalize $RUN_CONFIG]
set report_dir [file normalize $RUN_REPORT_DIR]
file mkdir $report_dir
set target_library [list $RUN_TARGET_LIBRARY]
set link_library [concat [list *] $target_library $RUN_SYNTHETIC_LIBRARY $RUN_LINK_LIBRARY]
read_verilog [file normalize $RUN_NETLIST]
current_design $RUN_TOP
link
source [file normalize $RUN_SDC]
# Activity is only annotated with power analysis already on.
set power_enable_analysis true
set power_model_preference nlpm
set opts [list]
if {$RUN_START_NS ne "" || $RUN_END_NS ne ""} {
  lappend opts -time [list "${RUN_START_NS}ns" "${RUN_END_NS}ns"]
}
set format [string tolower $RUN_ACTIVITY_FORMAT]
if {$format eq "fsdb"} {
  if {$RUN_STRIP_PATH ne ""} { read_fsdb {*}$opts -strip_path $RUN_STRIP_PATH [file normalize $RUN_ACTIVITY] } else { read_fsdb {*}$opts [file normalize $RUN_ACTIVITY] }
} elseif {$format eq "vcd"} {
  if {$RUN_STRIP_PATH ne ""} { read_vcd {*}$opts -strip_path $RUN_STRIP_PATH [file normalize $RUN_ACTIVITY] } else { read_vcd {*}$opts [file normalize $RUN_ACTIVITY] }
} elseif {$format eq "saif"} {
  if {$RUN_STRIP_PATH ne ""} { lappend opts -strip_path $RUN_STRIP_PATH }
  if {$RUN_INSTANCE ne ""} { lappend opts -path $RUN_INSTANCE }
  read_saif {*}$opts [file normalize $RUN_ACTIVITY]
} else { error "unsupported activity format: $format" }
# The simulation need not run at the SDC period (the bebop VCS harness clocks
# at 1 ns, UVM at the SDC period). The activity's own toggle rate on each clock
# source gives its simulation period; scale activity to the SDC clock.
set power_enable_clock_scaling true
foreach_in_collection clk [all_clocks] {
  set rate [lindex [lindex [get_switching_activity -toggle_rate [get_attribute $clk sources]] 0] 1]
  if {$rate <= 0} { error "the activity has no toggles on the source of clock [get_object_name $clk]" }
  set_power_clock_scaling -period [expr {2.0 / $rate}] $clk
}
update_power
# Report what the activity covers: the annotated instance, else the top.
if {$RUN_INSTANCE ne ""} { current_instance $RUN_INSTANCE }
report_switching_activity > [file join $report_dir switching_activity.rpt]
report_power -hierarchy -levels 8 > [file join $report_dir power_hierarchy.rpt]
report_power -verbose > [file join $report_dir power_total.rpt]
report_power -hierarchy -levels 2 -sort_by total_power > [file join $report_dir power_hierarchy_sorted.rpt]
exit
