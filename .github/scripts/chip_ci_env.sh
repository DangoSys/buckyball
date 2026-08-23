#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 <chip> <repo>" >&2
  exit 1
}

[[ $# -eq 2 ]] || usage

chip="$1"
repo="$2"
chip_dir="${repo}/examples/chips/${chip}"

if [[ ! -d "${chip_dir}" ]]; then
  echo "chip_ci_env: missing chip dir: ${chip_dir}" >&2
  exit 1
fi

verilator_config=""
chip_toml="${chip_dir}/chip.toml"
if [[ -f "${chip_toml}" ]]; then
  verilator_config="$(sed -n 's/^verilatorConfig[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "${chip_toml}" | head -n 1)"
fi

if [[ -f "${chip_dir}/regression/batch/bemu/workloads-elf.toml" ]]; then
  run_batch_tests=true
else
  run_batch_tests=false
fi

if [[ -f "${chip_dir}/regression/batch/bemu/workloads-elf-rushB.toml" ]]; then
  enable_rushb=true
  if [[ -z "${verilator_config}" ]]; then
    echo "chip_ci_env: rushB batch requires verilatorConfig in chip.toml" >&2
    exit 1
  fi
  rushb_verilator_config="${verilator_config%VerilatorConfig}RushBVerilatorConfig"
else
  enable_rushb=false
  rushb_verilator_config=""
fi

echo "verilator_config=${verilator_config}"
echo "rushb_verilator_config=${rushb_verilator_config}"
echo "enable_rushb=${enable_rushb}"
echo "run_batch_tests=${run_batch_tests}"
