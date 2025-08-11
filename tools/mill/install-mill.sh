#!/bin/bash

set -e

BBDIR=$(git rev-parse --show-toplevel)
MILL_DIR=$BBDIR/tools/mill

# Install Mill
curl -L https://raw.githubusercontent.com/lefou/millw/0.4.11/millw > $MILL_DIR/mill && chmod +x $MILL_DIR/mill

# Add Mill to PATH
# sudo mv mill /usr/local/bin/
echo "#mill path" >> ~/.${SHELL##*/}rc
echo "export PATH=\"${BBDIR}/tools/mill:\$PATH\"" >> ~/.${SHELL##*/}rc

# Verify installation
# mill --version
