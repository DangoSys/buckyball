#!/bin/bash

set -e

BBDIR=$(git rev-parse --show-toplevel)
MILL_DIR=$BBDIR/tools/mill

# Install Mill
curl -L https://raw.githubusercontent.com/lefou/millw/0.4.11/millw > $MILL_DIR/mill && chmod +x $MILL_DIR/mill

# Add Mill to PATH
source $BBDIR/scripts/utils.sh

export PATH="${BBDIR}/tools/mill:$PATH"
replace_content $BBDIR/env.sh "install-mill.sh" "export PATH=\"\${BBDIR}/tools/mill\:$PATH\""
# echo "#mill path" >> ~/.${SHELL##*/}rc
# echo "export PATH=\"${BBDIR}/tools/mill:\$PATH\"" >> ~/.${SHELL##*/}rc

# Verify installation
# mill --version
