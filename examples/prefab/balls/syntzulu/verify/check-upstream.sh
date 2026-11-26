#!/usr/bin/env bash
set -eu
verify=$(cd "$(dirname "$0")" && pwd)
src=$(cd "${1:?pass the upstream SYNtzulu checkout}/rtl/syntzulu" && pwd)
revision=$(git -C "$src" rev-parse HEAD)
test "$revision" = c9de5808376222c2de17e4f5771b4996b8b507fd
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
cd "$out"
sv2v -DBRAM --top=ReferenceCore "$src"/{BRAM_singlePort_readFirst,accumulator,adder_simd,bram_fifo,conv,integrator,integrator_and_fifo,layer_lp,neuron_lp,snn_lp,s2p}.sv \
  "$src"/{SPRAM_singlePort_readFirst,stack}.v "$verify/ReferenceCore.sv" > core.v
"${YOSYS:-yosys}" -Q -p 'read_verilog -defer core.v; hierarchy -top ReferenceCore; proc; opt; write_verilog -noattr net.v'
verilator --binary --timing -Wno-fatal --top-module reference_tb net.v "$verify/reference_tb.sv"
obj_dir/Vreference_tb
