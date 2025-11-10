#!/bin/bash

# Script to compile all Scala files under gemmini directory and log output

set -e

# Define paths
WORK_DIR="/home/daiyongyuan/buckyball/arch"
SRC_DIR="$WORK_DIR/src/main/scala/prototype/gemmini"
LOG_FILE="/home/daiyongyuan/buckyball/build_logs/gemmini_build.log"

# Ensure log directory exists
mkdir -p "/home/daiyongyuan/buckyball/build_logs"

# Clear or create the log file
> "$LOG_FILE"

echo "Starting compilation of Gemmini Scala files..." | tee -a "$LOG_FILE"

# Find all .scala files in gemmini subdirectories and compile them using sbt
find "$SRC_DIR" -type f -name "*.scala" | sort >> "$LOG_FILE" 2>&1

# Run sbt compile (assuming build.sbt is in root)
echo "Running sbt compile..." | tee -a "$LOG_FILE"
cd "$WORK_DIR" && sbt -J-Xmx8G -J-Xms2G -J-Xss4M -J-XX:+UseG1GC compile 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Compilation completed successfully." | tee -a "$LOG_FILE"
else
    echo "ERROR: Compilation failed. See $LOG_FILE for details." | tee -a "$LOG_FILE"
    exit 1
fi
