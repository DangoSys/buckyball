#!/bin/bash

set -e

ROOT="$(dirname "$(realpath "$0")")"

cd $ROOT
# git submodule update --init

$ROOT/spike/install-spike.sh
$ROOT/gem5/install-gem5.sh
