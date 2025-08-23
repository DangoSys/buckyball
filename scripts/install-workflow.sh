#!/bin/bash

# exit script if any command fails
set -e
set -o pipefail

BBDIR=$(git rev-parse --show-toplevel)

source ${BBDIR}/scripts/utils.sh

cd ${BBDIR}
replace_content ${BBDIR}/env.sh install-workflow "export PATH=${BBDIR}/workflow\:\$PATH"
# export PATH=${BBDIR}/workflow:\$PATH
source ${BBDIR}/env.sh

npx motia@latest create -n workflow -t python -y
cd ${BBDIR}/workflow/steps && rm *.{tsx,py}
# cd ${BBDIR}/workflow && npx motia dev -p 5000 
