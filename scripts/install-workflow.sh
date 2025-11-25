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

cd ${BBDIR}/workflow
npx motia@latest create -t python

pip install python-dotenv
pip install httpx

cd ${BBDIR}/workflow/steps && rm *.{py,json} || true
cd ${BBDIR}/workflow/steps && rm -r src/ || true
cd ${BBDIR}/workflow/steps && rm -r petstore/ || true
cd ${BBDIR}/workflow && rm -r services/ || true
cd ${BBDIR}/workflow && rm -r tutorial/ || true
cd ${BBDIR}/workflow && rm *.{md,tsx} || true
# cd ${BBDIR}/workflow && npx motia dev -p 5000

# install MCP
pip install mcp
pip install redis
pip install httpx_sse
