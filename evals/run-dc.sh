#!/bin/bash

# exit script if any command fails
set -e
set -o pipefail

KEEP_HIERARCHY=0

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
        --keep-hierarchy)
            KEEP_HIERARCHY=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --srcdir <design_source_directory> [--top <top_module>] [--keep-hierarchy]"
            echo "Example: $0 --srcdir arch/VecBall_1 --top VecBall_1 --keep-hierarchy"
            exit 1
            ;;
    esac
done

if [ -z "$SRC_DIR" ]; then
    echo "Error: Missing required parameter: srcdir"
    echo "Usage: $0 --srcdir <design_source_directory> [--top <top_module>] [--keep-hierarchy]"
    echo "Example: $0 --srcdir arch/VecBall_1 --top VecBall_1 --keep-hierarchy"
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
# Step0 执行build Verilator
#-------------------------------------------------------------------
# ${CYDIR}/voyager-test/scripts/build-verilator.sh --config ${CONFIG}

#-------------------------------------------------------------------
# Step1 搬运对应Config的Verilog到工作目录
#-------------------------------------------------------------------
DESIGN_SOURCE_DIR="${CYDIR}/${SRC_DIR}"
rm -rf ${DESIGN_DIR}/*
cp -r ${DESIGN_SOURCE_DIR}/* ${DESIGN_DIR}/

#-------------------------------------------------------------------
# Step2 替换SRAM
#-------------------------------------------------------------------
# echo "正在检查SRAM  File..."
# python ${CYDIR}/voyager-test/scripts/read_json.py $DB_FILE $DESIGN_DIR "/home/hxm123/tapeout-Voyager/sims/verilator/generated-src/chipyard.harness.TestHarness.GemminiRocketConfig/gen-collateral/metadata/seq_mems.json"

#-------------------------------------------------------------------
# Step3 编写tcl脚本
#-------------------------------------------------------------------


# 将文件路径格式化为 Tcl 所需的字符串（以空格分隔）
tcl_db_list=""

COMPILE_CMD="compile_ultra -retime -scan"
HIERARCHY_SETUP=""
if [ "$KEEP_HIERARCHY" -eq 1 ]; then
    COMPILE_CMD="compile_ultra -retime -scan -no_autoungroup"
    HIERARCHY_SETUP='
# Keep module boundaries for complete hierarchical reports.
set_ungroup [get_designs *] false
set_boundary_optimization [get_designs *] false
'
fi

cat > $TCL_FILE << EOF
# 设置搜索路径
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

# 读取设计文件
set file_list [glob -nocomplain -directory $DESIGN_DIR *.sv ]
analyze -format sverilog \$file_list
elaborate $TOP_MODULE

# 设置顶层模块名
set current_design "$TOP_MODULE"

# 链接设计
link

$HIERARCHY_SETUP

create_clock -name clock -period 2 [get_ports clock]

set_clock_uncertainty 0.6 [get_clocks clock]

set_input_delay 1.2 -clock clock  [remove_from_collection [all_inputs] [get_ports clock]]
set_output_delay 0.6 -clock clock [all_outputs]

#同频同相位

set_clock_equivalence clock

#输出负载

set_load 0.08 [all_outputs]

set_input_transition 0.2 [remove_from_collection [all_inputs] [get_ports clock]]

set_clock_transition 0.08 [get_clocks clock]


$COMPILE_CMD
write -format ddc -hierarchy -output $REPORT_DIR/design_compiled.ddc

# 生成报告
report_area -hierarchy -nosplit > $REPORT_DIR/area.rpt
report_hierarchy -noleaf > $REPORT_DIR/hierarchy.rpt
report_timing > $REPORT_DIR/timing.rpt
report_power -hierarchy > $REPORT_DIR/power.rpt

# 保存网表
write -format verilog -output $REPORT_DIR/netlist.v

# 退出
exit
EOF

#-------------------------------------------------------------------
# Step4 运行DC
#-------------------------------------------------------------------
echo "Running DC synthesis for design: ${CONFIG}, top module: $TOP_MODULE"
dc_shell -f $TCL_FILE

# rm $TCL_FILE
rm -rf ${CYDIR}/alib-52

echo "Synthesis completed. Reports are available in $REPORT_DIR directory."
