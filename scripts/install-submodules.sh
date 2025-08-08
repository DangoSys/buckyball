#!/usr/bin/env bash

# exit script if any command fails
set -e
set -o pipefail

BBDIR=$(git rev-parse --show-toplevel)

# git submodule update --init tools/motia
# git submodule update --init tools/vistools
# git submodule update --init thirdparty/chipyard

cd $BBDIR/tools/motia && ln -s ${BBDIR}/workflow ./steps
# npx motia@latest create --name bb-workflow --template default
npx motia dev -p 5000 
