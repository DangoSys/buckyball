#!/bin/bash

# Script to compile Chisel code using mill

set -e

# Define paths
WORK_DIR="/home/daiyongyuan/buckyball"
LOG_FILE="$WORK_DIR/build_logs/gemmini_build.log"

# Ensure log directory exists
mkdir -p "$WORK_DIR/build_logs"

# Clear or create the log file
> "$LOG_FILE"

echo "Starting Chisel compilation..." | tee -a "$LOG_FILE"

# Run mill compile (using mill instead of sbt for better performance)
echo "Running mill gemmini.compile..." | tee -a "$LOG_FILE"
cd "$WORK_DIR/arch" && mill -i gemmini.compile 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Compilation completed successfully." | tee -a "$LOG_FILE"
    exit 0
else
    echo "ERROR: Compilation failed. See $LOG_FILE for details." | tee -a "$LOG_FILE"
    exit 1
fi
