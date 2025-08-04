#!/bin/bash

BBDIR=$(git rev-parse --show-toplevel)
WORKFLOW_DIR=${BBDIR}/workflow
STEP_DIR=${BBDIR}/tools/motia/steps

# Check if workflow directory exists
if [ ! -d "$WORKFLOW_DIR" ]; then
  echo "Error: Workflow directory not found: $WORKFLOW_DIR"
  echo "Please running init.sh first"
  exit 1
fi

# Create target directory if it doesn't exist
mkdir -p "$STEP_DIR" || {
  echo "Error: Failed to create target directory: $STEP_DIR"
  exit 1
}

# Sync workflow content to target directory
echo "Syncing workflow files to $STEP_DIR..."
rsync -av --delete "$WORKFLOW_DIR/" "$STEP_DIR/" || {
  echo "Error: rsync failed"
  exit 1
}

echo "Sync completed successfully"
