#!/usr/bin/env bash
# Temporary parallel runner for a chip's Verilator manifest (removed after the migration).
# usage: tmp_run_manifest.sh <chip> <jobs> <out-dir>
set -u
chip=$1; jobs=$2; out=$3
root=/home/wanghui/Code/buckyball
manifest=$root/examples/chips/$chip/regression/batch/verilator/workloads-elf.toml
bin=$root/bebop/target/$chip/release/bebop
mkdir -p "$out"
: > "$out/summary.txt"
run_one() {
  name=$1
  elf=$(find "$root/bb-tests/output/$chip" -name "$name" -type f | head -1)
  dir="$out/$name"; mkdir -p "$dir"
  if [ -z "$elf" ]; then echo "$name MISSING" >> "$out/summary.txt"; return; fi
  timeout 1800 "$bin" run verilator --no-wave --elf "$elf" --log-dir "$dir" > "$dir/run.out" 2>&1
  code=$(grep -h "sim_exit" "$dir/stdout.log" 2>/dev/null | grep -oE "exit_code=[0-9]+" | head -1)
  cyc=$(tail -n1 "$dir/stderr.log" 2>/dev/null | grep -oE "C[0-9]+: +[0-9]+" | grep -oE "[0-9]+$")
  echo "$name ${code:-NO_EXIT} cycles=${cyc:-?}" >> "$out/summary.txt"
}
export -f run_one; export root chip bin out
grep -oE '"[^"]+"' "$manifest" | tr -d '"' | xargs -P "$jobs" -I{} bash -c 'run_one "$@"' _ {}
echo ALL_DONE >> "$out/summary.txt"
