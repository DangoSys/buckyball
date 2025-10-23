#!/bin/bash

set -e

BBDIR=$(git rev-parse --show-toplevel)
source ${BBDIR}/scripts/utils.sh

cd $BBDIR/sims/bebop
./scripts/install.sh
