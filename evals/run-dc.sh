#!/bin/bash

# exit script if any command fails
set -e
set -o pipefail


# DigitalTop
while [[ $# -gt 0 ]]; do
    case $1 in
        --srcdir)
            SRC_DIR="$2"
            shift 2
            ;;
        --top)
            TOP_MODULE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --srcdir <design_source_directory> --top [top_module](Optional)"
            echo "Example: $0 --srcdir arch/VecBall_1 --top VecBall_1"
            exit 1
            ;;
    esac
done

if [ -z "$SRC_DIR" ]; then
    echo "Error: Missing required parameter: srcdir"
    echo "Usage: $0 --srcdir <design_source_directory> --top [top_module](Optional)"
    echo "Example: $0 --srcdir arch/VecBall_1 --top VecBall_1"
    exit 1
fi



CYDIR=$(git rev-parse --show-toplevel)

WORK_DIR="${CYDIR}/bb-tests/output/dc"
DESIGN_DIR="${CYDIR}/bb-tests/output/dc/design"
REPORT_DIR="${CYDIR}/bb-tests/output/dc/reports"
TMP_DIR="${CYDIR}/bb-tests/output/dc/tmp"
TCL_FILE="${CYDIR}/bb-tests/output/dc/dc_script.tcl"
DB_FILE="/opt/dc/lib/TSMCHOME/SRAM_m4swbsoffg0p99v0c/"

mkdir -p $WORK_DIR
mkdir -p $DESIGN_DIR
mkdir -p $REPORT_DIR
mkdir -p $TMP_DIR

#-------------------------------------------------------------------
# Step0 Execute build Verilator
#-------------------------------------------------------------------
# ${CYDIR}/voyager-test/scripts/build-verilator.sh --config ${CONFIG}

#-------------------------------------------------------------------
# Step1 Copy Verilog files for corresponding Config to work directory
#-------------------------------------------------------------------
DESIGN_SOURCE_DIR="${CYDIR}/${SRC_DIR}"
rm -rf ${DESIGN_DIR}/*
cp -r ${DESIGN_SOURCE_DIR}/* ${DESIGN_DIR}/

#-------------------------------------------------------------------
# Step2 Replace SRAM
#-------------------------------------------------------------------
# echo "Checking SRAM File..."
# python ${CYDIR}/voyager-test/scripts/read_json.py $DB_FILE $DESIGN_DIR "/home/hxm123/tapeout-Voyager/sims/verilator/generated-src/chipyard.harness.TestHarness.GemminiRocketConfig/gen-collateral/metadata/seq_mems.json"

#-------------------------------------------------------------------
# Step3 Write tcl script
#-------------------------------------------------------------------


# Format file paths as Tcl required string (space-separated)
tcl_db_list=""


cat > $TCL_FILE << EOF
# Set search path
set search_path [list . $DESIGN_DIR]
define_design_lib work -path $TMP_DIR

set target_library "$tcl_db_list\
/data0/tools/lib/db/scc28nhkcp_hdc35p140_rvt_ffg_v0p99_0c_basic.db \
/data0/tools/lib/db/scc28nhkcp_hdc35p140_rvt_ffg_v0p99_0c_ccs.db \
/data0/tools/lib/db/scc28nhkcp_hdc35p140_rvt_ffg_v0p99_0c_ecsm.db \
"
set link_library "$tcl_db_list\
/data0/tools/lib/db/scc28nhkcp_hdc35p140_rvt_ffg_v0p99_0c_basic.db \
/data0/tools/lib/db/scc28nhkcp_hdc35p140_rvt_ffg_v0p99_0c_ccs.db \
/data0/tools/lib/db/scc28nhkcp_hdc35p140_rvt_ffg_v0p99_0c_ecsm.db \
"

# Read design files
set file_list [glob -nocomplain -directory $DESIGN_DIR *.sv ]
analyze -format sverilog \$file_list
elaborate $TOP_MODULE

# Set top module name
set current_design "$TOP_MODULE"

# Link design
link

create_clock -name clk1 -period 2 [get_ports clock]

set_clock_uncertainty 0.6 [get_clocks clock]

set_input_delay 1.2 -clock clk1  [remove_from_collection [all_inputs] [get_ports clock]]
set_output_delay 0.6 -clock clk1 [all_outputs]

# Same frequency same phase

set_clock_equivalence clk1

# Output load

set_load 0.08 [all_outputs]

set_input_transition 0.2 [remove_from_collection [all_inputs] [get_ports clock]]

set_clock_transition 0.08 [get_clocks clk1]



compile_ultra -scan
write -format ddc -hierarchy -output $REPORT_DIR/design_compiled.ddc

# Generate reports
report_area -hierarchy -nosplit > $REPORT_DIR/area.rpt
report_timing > $REPORT_DIR/timing.rpt
report_power -hierarchy > $REPORT_DIR/power.rpt

# Save netlist
write -format verilog -output $REPORT_DIR/netlist.v

# Exit
exit
EOF

#-------------------------------------------------------------------
# Step4 Run DC
#-------------------------------------------------------------------
echo "Running DC synthesis for design: ${CONFIG}, top module: $TOP_MODULE"
dc_shell -f $TCL_FILE

# rm $TCL_FILE
rm -rf ${CYDIR}/alib-52

echo "Synthesis completed. Reports are available in $REPORT_DIR directory."
