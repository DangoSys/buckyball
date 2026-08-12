# Shared PrimeTime PX implementation. Chip tapeout/power.tcl files are thin
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
read_sdc [file normalize $RUN_SDC]
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
  read_saif {*}$opts [file normalize $RUN_ACTIVITY]
} else { error "unsupported activity format: $format" }
set power_enable_analysis true
set power_model_preference nlpm
update_power
report_power -hierarchy -levels 3 > [file join $report_dir power_hierarchy.rpt]
report_power -verbose > [file join $report_dir power_total.rpt]
report_power -hierarchy -levels 2 -sort_by total_power > [file join $report_dir power_hierarchy_sorted.rpt]
exit
