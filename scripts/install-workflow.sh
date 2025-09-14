#!/bin/bash

# exit script if any command fails
# set -e
# set -o pipefail

BBDIR=$(git rev-parse --show-toplevel)

source ${BBDIR}/scripts/utils.sh

cd ${BBDIR}
replace_content ${BBDIR}/env.sh install-workflow "export PATH=${BBDIR}/workflow:\$PATH"
# export PATH=${BBDIR}/workflow:\$PATH
source ${BBDIR}/env.sh

# lower node veersion is not supported for motia
conda install -c conda-forge nodejs=20 -y


cd ${BBDIR}/workflow
ln -s ${CONDA_PREFIX} ./python_modules

cd ${BBDIR}
npx motia@latest create -n workflow -t python

pip install python-dotenv
pip install httpx

cd ${BBDIR}/workflow/steps && rm *.{tsx,py,json}
cd ${BBDIR}/workflow && rm -r services/
cd ${BBDIR}/workflow && rm *.{tsx,py,md}
# cd ${BBDIR}/workflow && npx motia dev -p 5000
