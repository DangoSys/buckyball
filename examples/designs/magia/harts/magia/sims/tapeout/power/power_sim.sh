#!/usr/bin/env bash
# The switching activity of the magia hart alone: its single-hart system runs
# WORKLOAD on the entry's Verilator runner (VCS does not support the
# multi-dimensional interface arrays of the MAGIA sources); the FST of the run
# becomes a VCD, then the SAIF PrimeTime reads. The hart is
# TOP/BBSimHarness/soc/system/hart_MagiaHart there and the synthesis top
# itself, so the strip path is that instance.
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

# The workload is <output>/workloads/..., <output> the simulation's output.
SIMULATION=${WORKLOAD%%/workloads/*}
RUNNER="$SIMULATION/bin/bebop-verilator"
if [[ ! -x "$RUNNER" ]]; then
  echo "missing the Verilator runner $RUNNER; run bbdev bebop-verilator --build first" >&2
  exit 2
fi
RUN="$OUTPUT_DIR/verilator"
mkdir -p "$RUN"
"$RUNNER" --elf "$WORKLOAD" --log-dir "$RUN"
# The VCD is tens of GBs: it streams through a pipe into vcd2saif. vcd2saif
# reads only VCD's own variable types, so SystemVerilog ones become wires.
mkfifo "$RUN/activity.vcd"
fst2vcd -f "$RUN/waveform/waveform.fst" |
  sed -E 's/^\$var (logic|bit|byte|shortint|int|longint|enum) /$var wire /' > "$RUN/activity.vcd" &
vcd2saif -input "$RUN/activity.vcd" -output "$ACTIVITY_FILE"
wait $!
rm "$RUN/activity.vcd" "$RUN/waveform/waveform.fst"
