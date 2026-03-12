# Pegasus Synthesis-Only Script
# Assumes IP cores already created (pegasus_proj exists with IPs synthesized)
#
# Usage (from vivado/ directory):
#   vivado -mode batch -source synth_only.tcl

set PART        xcu280-fsvh2892-2L-e
set TOP         PegasusHarness
set PROJ_NAME   pegasus_proj
set PROJ_DIR    ./${PROJ_NAME}
set GEN_DIR     ./generated

# Open existing project
open_project ${PROJ_DIR}/${PROJ_NAME}.xpr

# Replace user RTL sources (keep IP files intact)
set old_rtl [get_files -quiet -filter {NAME =~ "*/generated/*"}]
if {[llength $old_rtl] > 0} {
  remove_files $old_rtl
}

# Patterns for simulation-only files
proc is_sim_only {fname} {
  set base [file tail $fname]
  foreach pat {BackdoorGet* BackdoorPut* *DPI.v *DPI.sv ClockSource* EICG_wrapper* GenericDigital*IOCell*} {
    if {[string match $pat $base]} { return 1 }
  }
  return 0
}

set synth_files {}
foreach f [concat [glob -nocomplain ${GEN_DIR}/*.sv] [glob -nocomplain ${GEN_DIR}/*.v]] {
  if {![is_sim_only $f]} {
    lappend synth_files $f
  } else {
    puts "  skipping sim-only: [file tail $f]"
  }
}
if {[llength $synth_files] > 0} { add_files $synth_files }
set_property top ${TOP} [current_fileset]

# Lock IP OOC runs so synth_design uses their DCPs directly
foreach run [get_runs -filter {IS_SYNTHESIS == 1 && NAME != "synth_1"}] {
  set_property IS_ENABLED false $run
}

set_param general.maxThreads 32

puts "==> Running synthesis..."
synth_design -top ${TOP} -part ${PART} \
  -flatten_hierarchy rebuilt \
  -gated_clock_conversion auto \
  -fsm_extraction one_hot \
  -directive RuntimeOptimized

write_checkpoint -force ${PROJ_DIR}/post_synth.dcp
report_utilization -file ${PROJ_DIR}/utilization_synth.rpt
puts "==> Synthesis complete: ${PROJ_DIR}/post_synth.dcp"
