#!/usr/bin/env bash

# exit script if any command fails
set -e
set -o pipefail

BBDIR=$(git rev-parse --show-toplevel)

# git submodule update --init tools/motia
# git submodule update --init tools/vistools
# git submodule update --init thirdparty/chipyard

cd ${BBDIR}
npx motia@latest create -n workflow -t python -y
cd ${BBDIR}/workflow/steps && rm *.{tsx,py}
# cd ${BBDIR}/workflow && npx motia dev -p 5000 
