#!/bin/bash

# $1 is the simulator binary
# $2 is the workload elf
# $3 is the stdout file
# $4 is the disasm file
# $5 is the batch flag
# $6 is the VCD file path
# $7 is the log file path

# Check if VCD path is provided and not empty
if [ -z "$6" ]; then
  echo "Error: VCD file path is required but not provided"
  exit 1
fi

# Check if log path is provided and not empty
if [ -z "$7" ]; then
  echo "Error: Log file path is required but not provided"
  exit 1
fi

if [ $5 == "True" ]; then
  $1 +permissive +loadmem=$2 +loadmem_addr=800000000 +batch +vcd=$6 +log=$7 +permissive-off $2 > >(tee $3) 2> >(spike-dasm > $4)
else
  $1 +permissive +loadmem=$2 +loadmem_addr=800000000 +vcd=$6 +log=$7 +permissive-off $2 > >(tee $3) 2> >(spike-dasm > $4)
fi
