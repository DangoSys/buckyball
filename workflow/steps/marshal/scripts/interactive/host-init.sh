#!/bin/bash

# This script will run on the host from the workload directory
# (e.g. workloads/example-fed) every time the workload is built.
# It is recommended to call into something like a makefile because
# this script may be called multiple times.

BBDIR=$(git rev-parse --show-toplevel)

cd $BBDIR && source env.sh

echo "Building marshal workload"
bbdev workload --build

mkdir -p $BBDIR/workflow/steps/marshal/output
rm -rf $BBDIR/workflow/steps/marshal/output/overlay
mkdir -p $BBDIR/workflow/steps/marshal/output/overlay/root

# Copy workload binaries to /root directory
cp -r $BBDIR/bb-tests/output/workloads/src/* $BBDIR/workflow/steps/marshal/output/overlay/root/
