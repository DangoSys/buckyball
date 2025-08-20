#!/usr/bin/env bash

# exit script if any command fails
set -e
set -o pipefail

BBDIR=$(git rev-parse --show-toplevel)

source ${BBDIR}/scripts/utils.sh

# git submodule update --init tools/motia
# git submodule update --init tools/vistools
# git submodule update --init thirdparty/chipyard

cd ${BBDIR}
export PATH="${BBDIR}/workflow:\$PATH"
replace_content $BBDIR/env.sh "install-workflow.sh" "export PATH=\"\${BBDIR}/workflow\:\$PATH\""

npx motia@latest create -n workflow -t python -y
cd ${BBDIR}/workflow/steps && rm *.{tsx,py}
# cd ${BBDIR}/workflow && npx motia dev -p 5000 
