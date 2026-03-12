# Pegasus Vivado Build Script
# Synthesizes and implements the Pegasus FPGA design for AU280
#
# Usage:
#   vivado -mode batch -source build.tcl
#
# Prerequisites:
#   1. Run ElaboratePegasus to generate Verilog:
#      cd arch && sbt "runMain sims.pegasus.ElaboratePegasus"
#      Then copy generated/*.sv to ./generated/
#   2. Vivado 2021.1+ must be installed and in PATH

set PART        xcu280-fsvh2892-2L-e
set TOP         PegasusHarness
set PROJ_NAME   pegasus_proj
set PROJ_DIR    ./${PROJ_NAME}
set GEN_DIR     ./generated
set IP_DIR      ./ip
set XDC_FILE    ./au280.xdc

################################################################################
# Step 1: Create project
################################################################################

create_project ${PROJ_NAME} ${PROJ_DIR} -part ${PART} -force

# Set project properties
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

################################################################################
# Step 2: Create IPs
################################################################################

puts "==> Creating IP cores..."
source ${IP_DIR}/xdma_ip.tcl
source ${IP_DIR}/hbm2_ip.tcl
source ${IP_DIR}/axi_crossbar_ip.tcl

# Synthesize IPs out-of-context so their netlists are available to synth_design
puts "==> Synthesizing IP cores (OOC)..."
set top_ips [list xdma_0 hbm_0 axi_crossbar_0 axi_dwidth_converter_0]
foreach ip_name $top_ips {
  set ip [get_ips $ip_name -quiet]
  if {$ip ne ""} {
    puts "  synth_ip: ${ip_name}"
    synth_ip $ip
  }
}

# After synth_ip, exclude unisim/secureip library files from top-level synthesis
foreach f [get_files -filter {FILE_TYPE == "Verilog" && NAME =~ "*/unisim*"}] {
  set_property USED_IN_SYNTHESIS false $f
}
foreach f [get_files -filter {FILE_TYPE == "Verilog" && NAME =~ "*/secureip*"}] {
  set_property USED_IN_SYNTHESIS false $f
}
# Also exclude unisim_comp.v which is in scripts/rt/data/
foreach f [get_files -filter {NAME =~ "*/scripts/rt/data/*"}] {
  set_property USED_IN_SYNTHESIS false $f
}

################################################################################
# Step 3: Add generated Verilog sources
# Exclude simulation-only files: DPI-C wrappers, clock generators, IO cell models
################################################################################

# Patterns for simulation-only files to exclude from synthesis
set sim_only_patterns {
  BackdoorGet* BackdoorPut*
  *DPI.v *DPI.sv
  ClockSource* EICG_wrapper*
  GenericDigital*IOCell*
}

proc is_sim_only {fname patterns} {
  set base [file tail $fname]
  foreach pat $patterns {
    if {[string match $pat $base]} { return 1 }
  }
  return 0
}

puts "==> Adding Verilog sources (excluding simulation-only files)..."

set sv_files [glob -nocomplain ${GEN_DIR}/*.sv]
set v_files  [glob -nocomplain ${GEN_DIR}/*.v]

set synth_files {}
foreach f [concat $sv_files $v_files] {
  if {![is_sim_only $f $sim_only_patterns]} {
    lappend synth_files $f
  } else {
    puts "  skipping sim-only: [file tail $f]"
  }
}

if {[llength $synth_files] > 0} {
  add_files $synth_files
}

set_property top ${TOP} [current_fileset]

################################################################################
# Step 4: Add constraints
################################################################################

puts "==> Adding constraints..."
add_files -fileset constrs_1 ${XDC_FILE}

################################################################################
# Step 5: Synthesis
################################################################################

# Multi-threading + runtime-optimized strategy (256-core server)
set_param general.maxThreads 32

puts "==> Running synthesis..."
synth_design -top ${TOP} -part ${PART} \
  -flatten_hierarchy rebuilt \
  -gated_clock_conversion auto \
  -fsm_extraction one_hot \
  -directive RuntimeOptimized

write_checkpoint -force ${PROJ_DIR}/post_synth.dcp
report_utilization -file ${PROJ_DIR}/utilization_synth.rpt

################################################################################
# Step 6: Implementation
################################################################################

puts "==> Optimizing design..."
opt_design   -directive RuntimeOptimized

puts "==> Placing design..."
place_design -directive RuntimeOptimized

puts "==> Routing design..."
route_design -directive RuntimeOptimized

# Post-implementation reports
write_checkpoint -force ${PROJ_DIR}/post_route.dcp
report_timing_summary -file ${PROJ_DIR}/timing_summary.rpt -warn_on_violation
report_utilization -file ${PROJ_DIR}/utilization_impl.rpt
report_power -file ${PROJ_DIR}/power.rpt

# Check timing
if {[get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]] < 0} {
  puts "WARNING: Setup timing violations detected. Check timing_summary.rpt"
  puts "WARNING: You may need to reduce PegasusConfig frequency and re-elaborate."
} else {
  puts "INFO: Timing closure achieved."
}

################################################################################
# Step 7: Generate bitstream
################################################################################

puts "==> Generating bitstream..."
write_bitstream -force ${PROJ_DIR}/pegasus.bit

puts "==> Build complete: ${PROJ_DIR}/pegasus.bit"
puts ""
puts "Next steps:"
puts "  1. Check timing_summary.rpt for actual Fmax"
puts "  2. If Fmax differs from 200 MHz, update PegasusConfig frequencies and re-elaborate"
puts "  3. Program FPGA: source scripts/program_fpga.sh"
puts "  4. Install XDMA driver: nix run .#install-xdma"
puts "  5. Run workload: pegasus-run ./workload.elf"
