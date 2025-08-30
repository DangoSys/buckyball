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

#============================================================
# install PySocks dependency for users using proxy, let them Seamless installation
# Save original proxy settings
echo "Saving original proxy settings..."
ORIGINAL_HTTP_PROXY="${http_proxy:-}"
ORIGINAL_HTTPS_PROXY="${https_proxy:-}"
ORIGINAL_ALL_PROXY="${all_proxy:-}"
ORIGINAL_HTTP_PROXY="${HTTP_PROXY:-}"
ORIGINAL_HTTPS_PROXY="${HTTPS_PROXY:-}"
ORIGINAL_ALL_PROXY="${ALL_PROXY:-}"

# Temporarily disable proxy to avoid SOCKS dependency issues
echo "Temporarily disabling proxy for installation..."
export http_proxy=""
export https_proxy=""
export all_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""
export ALL_PROXY=""

echo "Installing PySocks dependency..."
${BBDIR}/workflow/python_modules/bin/python -m pip install pysocks

# Restore original proxy settings
echo "Restoring original proxy settings..."
export http_proxy="${ORIGINAL_HTTP_PROXY}"
export https_proxy="${ORIGINAL_HTTPS_PROXY}"
export all_proxy="${ORIGINAL_ALL_PROXY}"
export HTTP_PROXY="${ORIGINAL_HTTP_PROXY}"
export HTTPS_PROXY="${ORIGINAL_HTTPS_PROXY}"
export ALL_PROXY="${ORIGINAL_ALL_PROXY}"
#============================================================

npx motia@latest create -n workflow -t python -y
cd ${BBDIR}/workflow/steps && rm *.{tsx,py,json}
cd ${BBDIR}/workflow/steps && rm -r services/
# cd ${BBDIR}/workflow && npx motia dev -p 5000 
