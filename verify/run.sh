#!/bin/bash

VERIFY_WORKSPACE="$(dirname "$(realpath "$0")")"

bbdev verilator --verilog "--balltype vecball --output_dir $VERIFY_WORKSPACE/dut/"
