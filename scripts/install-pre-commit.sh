#!/usr/bin/env bash
set -e

BBDIR=$(git rev-parse --show-toplevel)
source ${BBDIR}/scripts/utils.sh

# Check if scalafmt is already installed
if command -v scalafmt &> /dev/null; then
    echo "scalafmt is already installed"
    scalafmt --version
    exit 0
fi

# If not, install coursier and scalafmt
if ! command -v cs &> /dev/null; then
    echo "Installing coursier..."
    curl -fL https://github.com/coursier/launchers/raw/master/cs-x86_64-pc-linux.gz | gzip -d > cs
    chmod +x cs && ./cs setup --yes
    rm -f cs
fi

replace_content ${BBDIR}/env.sh install-pre-commit "export PATH=$HOME/.local/share/coursier/bin:\$PATH"

# Install scalafmt
echo "Installing scalafmt..."
cs install scalafmt

echo "Installation complete!"
echo "scalafmt version:"
scalafmt --version
