#!/bin/bash

# exit script if any command fails
set -e
set -o pipefail

BBDIR=$(git rev-parse --show-toplevel)

source ${BBDIR}/scripts/utils.sh

cd ${BBDIR}
replace_content ${BBDIR}/env.sh install-workflow "export PATH=${BBDIR}/workflow:\$PATH"
source ${BBDIR}/env.sh

# lower node veersion is not supported for motia
conda install -c conda-forge nodejs=20 -y
# conda install -c conda-forge redis -y
pip install python-dotenv
pip install httpx

cd ${BBDIR}/workflow
ln -s ${CONDA_PREFIX} ./python_modules || true

cd ${BBDIR}/workflow
# if package.json does not exist, create and install motia
# if [ ! -f package.json ]; then
  # install system dependencies for compiling Redis (redis-memory-server requires)
#   if command -v apt-get &> /dev/null; then
#     sudo apt-get update -qq
#     sudo apt-get install -y libsystemd-dev build-essential || true
#   fi
#   npm init -y
#   npm install motia@0.13.0-beta.161
# fi
export USE_SYSTEMD=no
npm init -y
npm install motia@0.13.0-beta.161
npx motia create -t python

cd ${BBDIR}/workflow/steps && rm *.{py,json} || true
cd ${BBDIR}/workflow/steps && rm -r src/ || true
cd ${BBDIR}/workflow/steps && rm -r petstore/ || true
cd ${BBDIR}/workflow && rm -r src/ || true
cd ${BBDIR}/workflow && rm -r tutorial/ || true
cd ${BBDIR}/workflow && rm *.{md,tsx,rdb} || true

# install MCP
pip install mcp
pip install redis
pip install httpx_sse
