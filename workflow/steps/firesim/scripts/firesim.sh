#!/bin/bash

# $1 is the firesim cmd

# Check if the firesim path is provided and not empty
if [ "$1" != "" ]; then
    echo "Usage: $0"
    exit 0
fi

FIRESIM_CMD=$1 
YAML_PATH=$(dirname "${BASH_SOURCE[0]}")

echo $YAML_PATH

$FIRESIM_CMD \
  -a $YAML_PATH/config_hwdb.yaml \
  -b $YAML_PATH/config_build.yaml \
  -r $YAML_PATH/config_build_recipes.yaml \
  -c $YAML_PATH/config_runtime.yaml \
