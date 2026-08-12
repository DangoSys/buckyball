#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then echo "usage: $0 <run.env>" >&2; exit 2; fi
source "$1"
: "${ACTIVITY_FILE:?run.env missing ACTIVITY_FILE}"
: "${ACTIVITY_FORMAT:?run.env missing ACTIVITY_FORMAT}"
: "${NETLIST:?run.env missing NETLIST}"
: "${OUTPUT_DIR:?run.env missing OUTPUT_DIR}"
: "${WORKLOAD:?run.env missing WORKLOAD; pass --workload or set tapeout.power_workload}"

if [[ "$ACTIVITY_FORMAT" != "vcd" ]]; then
  echo "Pebble gate power simulation currently emits VCD, got $ACTIVITY_FORMAT" >&2
  exit 2
fi
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
RTL_DIR="$ROOT/arch/build/${CONFIG}"
MEM_CONF="$RTL_DIR/mems.conf"
BUILD="$OUTPUT_DIR/gate-vcs"
mkdir -p "$BUILD"
if [[ ! -f "$MEM_CONF" ]]; then echo "missing FIRRTL memory manifest: $MEM_CONF" >&2; exit 2; fi
python3 "$ROOT/bbdev/api/steps/dc/scripts/emit_seq_mem_models.py" --mem-conf "$MEM_CONF" --output "$BUILD/seq_mem_models.sv"

cat > "$BUILD/GatePowerHarness.sv" <<'SV'
`timescale 1ns/1ps
module GatePowerHarness;
  reg clock = 1'b0;
  reg reset = 1'b1;
  DigitalTop dut(
    .auto_chipyard_prcictrl_domain_reset_setter_clock_in_member_allClocks_uncore_clock(clock),
    .auto_chipyard_prcictrl_domain_reset_setter_clock_in_member_allClocks_uncore_reset(reset),
    .mem_axi4_0_aw_ready(axi_aw_ready), .mem_axi4_0_aw_valid(axi_aw_valid),
    .mem_axi4_0_aw_bits_id(axi_aw_id), .mem_axi4_0_aw_bits_addr(axi_aw_addr),
    .mem_axi4_0_aw_bits_len(axi_aw_len), .mem_axi4_0_aw_bits_size(axi_aw_size),
    .mem_axi4_0_aw_bits_burst(axi_aw_burst), .mem_axi4_0_aw_bits_lock(axi_aw_lock),
    .mem_axi4_0_aw_bits_cache(axi_aw_cache), .mem_axi4_0_aw_bits_prot(axi_aw_prot),
    .mem_axi4_0_aw_bits_qos(axi_aw_qos), .mem_axi4_0_w_ready(axi_w_ready),
    .mem_axi4_0_w_valid(axi_w_valid), .mem_axi4_0_w_bits_data(axi_w_data),
    .mem_axi4_0_w_bits_strb(axi_w_strb), .mem_axi4_0_w_bits_last(axi_w_last),
    .mem_axi4_0_b_ready(axi_b_ready), .mem_axi4_0_b_valid(axi_b_valid),
    .mem_axi4_0_b_bits_id(axi_b_id), .mem_axi4_0_b_bits_resp(axi_b_resp),
    .mem_axi4_0_ar_ready(axi_ar_ready), .mem_axi4_0_ar_valid(axi_ar_valid),
    .mem_axi4_0_ar_bits_id(axi_ar_id), .mem_axi4_0_ar_bits_addr(axi_ar_addr),
    .mem_axi4_0_ar_bits_len(axi_ar_len), .mem_axi4_0_ar_bits_size(axi_ar_size),
    .mem_axi4_0_ar_bits_burst(axi_ar_burst), .mem_axi4_0_ar_bits_lock(axi_ar_lock),
    .mem_axi4_0_ar_bits_cache(axi_ar_cache), .mem_axi4_0_ar_bits_prot(axi_ar_prot),
    .mem_axi4_0_ar_bits_qos(axi_ar_qos), .mem_axi4_0_r_ready(axi_r_ready),
    .mem_axi4_0_r_valid(axi_r_valid), .mem_axi4_0_r_bits_id(axi_r_id),
    .mem_axi4_0_r_bits_data(axi_r_data), .mem_axi4_0_r_bits_resp(axi_r_resp),
    .mem_axi4_0_r_bits_last(axi_r_last), .interrupts(1'b0),
    .auto_mbus_fixedClockNode_anon_out_clock(), .auto_mbus_fixedClockNode_anon_out_reset());
  wire axi_aw_ready, axi_aw_valid, axi_aw_lock, axi_w_ready, axi_w_valid, axi_w_last, axi_b_ready, axi_b_valid;
  wire axi_ar_ready, axi_ar_valid, axi_ar_lock, axi_r_ready, axi_r_valid, axi_r_last;
  wire [3:0] axi_aw_id, axi_b_id, axi_ar_id, axi_r_id; wire [1:0] axi_b_resp, axi_r_resp, axi_aw_burst, axi_ar_burst;
  wire [31:0] axi_aw_addr, axi_ar_addr; wire [7:0] axi_aw_len, axi_ar_len, axi_w_strb;
  wire [2:0] axi_aw_size, axi_aw_prot, axi_ar_size, axi_ar_prot; wire [3:0] axi_aw_cache, axi_aw_qos, axi_ar_cache, axi_ar_qos;
  wire [63:0] axi_w_data, axi_r_data;
  BBSimDRAM #(.ADDR_BITS(32), .CHIP_ID(0), .CLOCK_HZ(100000000), .DATA_BITS(64), .ID_BITS(4), .LINE_SIZE(64), .MEM_BASE(40'd2147483648), .MEM_SIZE(268435456)) mem(.clock(clock), .reset(reset), .axi_aw_ready(axi_aw_ready), .axi_aw_valid(axi_aw_valid), .axi_aw_bits_id(axi_aw_id), .axi_aw_bits_addr(axi_aw_addr), .axi_aw_bits_len(axi_aw_len), .axi_aw_bits_size(axi_aw_size), .axi_aw_bits_burst(axi_aw_burst), .axi_aw_bits_lock(axi_aw_lock), .axi_aw_bits_cache(axi_aw_cache), .axi_aw_bits_prot(axi_aw_prot), .axi_aw_bits_qos(axi_aw_qos), .axi_w_ready(axi_w_ready), .axi_w_valid(axi_w_valid), .axi_w_bits_data(axi_w_data), .axi_w_bits_strb(axi_w_strb), .axi_w_bits_last(axi_w_last), .axi_b_ready(axi_b_ready), .axi_b_valid(axi_b_valid), .axi_b_bits_id(axi_b_id), .axi_b_bits_resp(axi_b_resp), .axi_ar_ready(axi_ar_ready), .axi_ar_valid(axi_ar_valid), .axi_ar_bits_id(axi_ar_id), .axi_ar_bits_addr(axi_ar_addr), .axi_ar_bits_len(axi_ar_len), .axi_ar_bits_size(axi_ar_size), .axi_ar_bits_burst(axi_ar_burst), .axi_ar_bits_lock(axi_ar_lock), .axi_ar_bits_cache(axi_ar_cache), .axi_ar_bits_prot(axi_ar_prot), .axi_ar_bits_qos(axi_ar_qos), .axi_r_ready(axi_r_ready), .axi_r_valid(axi_r_valid), .axi_r_bits_id(axi_r_id), .axi_r_bits_data(axi_r_data), .axi_r_bits_resp(axi_r_resp), .axi_r_bits_last(axi_r_last));
  always #5 clock = ~clock;
  initial begin
    longint timeout_ns;
    if (!$value$plusargs("timeout-ns=%d", timeout_ns)) timeout_ns = 100000;
    $dumpfile("activity.vcd"); $dumpvars(0, dut);
    repeat (10) @(posedge clock); reset=0; #(timeout_ns); $finish;
  end
endmodule
SV

CELL_MODEL="$(dirname "$(dirname "$(dirname "${TARGET_LIBRARY:?missing TARGET_LIBRARY}" )")")/verilog/scc018ug_uhd_rvt.v"
if [[ ! -f "$CELL_MODEL" ]]; then echo "missing standard-cell Verilog model: $CELL_MODEL" >&2; exit 2; fi

VCS_CMD=(vcs -full64 -sverilog -timescale=1ns/1ps -top GatePowerHarness -debug_access+all -hsopt=off -Mdir="$BUILD/csrc" -o "$BUILD/simv" -l "$BUILD/compile.log" -CFLAGS "-std=c++17 -I$ROOT/result/include -I$RTL_DIR -I$ROOT/arch/src/csrc/include" -LDFLAGS "-ldramsim3 -lz -lstdc++ -L$ROOT/result/lib -Wl,-rpath,$ROOT/result/lib")
mapfile -t DPI_MODELS < <(find "$RTL_DIR" -maxdepth 1 -name '*DPI.v' -print | sort)
VCS_CMD+=("$NETLIST" "$CELL_MODEL" "$BUILD/seq_mem_models.sv" "$RTL_DIR/BBSimDRAM.v" "${DPI_MODELS[@]}" "$BUILD/GatePowerHarness.sv" "$ROOT/arch/src/csrc/src/monitor/ioe/BBSimDRAM.cc" "$ROOT/arch/src/csrc/src/monitor/ioe/mm.cc" "$ROOT/arch/src/csrc/src/monitor/ioe/mm_dramsim3.cc" "$ROOT/bbdev/api/steps/dc/scripts/gate_dpi.cc")
"${VCS_CMD[@]}"
(cd "$BUILD" && env -u LD_LIBRARY_PATH ./simv +batch +vcd "+elf=$WORKLOAD" "+dramsim_ini_dir=$ROOT/result/share/dramsim3/configs" "+timeout-ns=${END_NS:-100000000}")
cp "$BUILD/activity.vcd" "$ACTIVITY_FILE"
