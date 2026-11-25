#!/usr/bin/env bash
# The switching activity of a hart alone: its single-hart system
# (BBSimHarness, the RTL of `bbdev bebop-verilator --verilog "--hart <h>"`)
# runs WORKLOAD on VCS; the VCD of the run becomes the SAIF PrimeTime reads.
# The hart is BBSimVcsHarness/dut/soc/system/hart_<Top> there and the
# synthesis top itself, so the strip path is that instance.
set -euo pipefail

if [[ $# != 1 ]]; then echo "usage: $0 <run.env>" >&2; exit 2; fi
source "$1"
: "${ACTIVITY_FILE:?run.env missing ACTIVITY_FILE}"
: "${ACTIVITY_FORMAT:?run.env missing ACTIVITY_FORMAT}"
: "${OUTPUT_DIR:?run.env missing OUTPUT_DIR}"
: "${WORKLOAD:?run.env missing WORKLOAD; pass --workload <built ELF name>}"
if [[ "$ACTIVITY_FORMAT" != "saif" ]]; then
  echo "hart power simulation emits SAIF, got $ACTIVITY_FORMAT" >&2
  exit 2
fi

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ENTRY=$(cd "$(dirname "$0")/../.." && pwd)
# The workload is <output>/workloads/..., <output> the simulation's output.
SIMULATION=${WORKLOAD%%/workloads/*}
RTL="$SIMULATION/arch/bebop/BBSimHarness"
if [[ ! -f "$RTL/BBSimHarness.sv" ]]; then
  echo "missing the hart system RTL $RTL; run bbdev bebop-verilator --verilog first" >&2
  exit 2
fi
TARGET="$SIMULATION/emu/vcs/target"
BUILD="$OUTPUT_DIR/vcs"
cargo build --release --manifest-path "$ENTRY/emu/Cargo.toml" --target-dir "$TARGET" --bin bebop
"$TARGET/release/bebop" build vcs --rtl-dir "$RTL" --artifact-dir "$BUILD" --jobs 16
mkdir -p "$BUILD/run/dramsim"
(cd "$BUILD/run" && env -u LD_LIBRARY_PATH "$BUILD/simv" +permissive "+elf=$WORKLOAD" +batch \
  "+dramsim_ini_dir=$ROOT/result/share/dramsim3/configs" "+dramsim_outdir=$BUILD/run/dramsim" \
  "+vcd=$BUILD/run/activity.vcd" +permissive-off)
vcd2saif -input "$BUILD/run/activity.vcd" -output "$ACTIVITY_FILE"
# The VCD is GBs; the SAIF holds what PrimeTime reads.
rm "$BUILD/run/activity.vcd"
