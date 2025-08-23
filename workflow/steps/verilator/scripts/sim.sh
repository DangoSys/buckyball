#!/bin/bash

# $1 is the simulator binary
# $2 is the workload elf
# $3 is the stdout file
# $4 is the disasm file
# $5 is the batch flag

if [ $5 == "True" ]; then
  $1 +permissive +loadmem=$2 +loadmem_addr=800000000 --batch +permissive-off $2 > >(tee $3) 2> >(spike-dasm > $4)
else
  $1 +permissive +loadmem=$2 +loadmem_addr=800000000 +permissive-off $2 > >(tee $3) 2> >(spike-dasm > $4)
fi
