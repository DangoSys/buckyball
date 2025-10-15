#!/bin/bash

set -e

BBDIR=$(git rev-parse --show-toplevel)
source ${BBDIR}/scripts/utils.sh

# install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

replace_content ${BBDIR}/env.sh bb-doc-server "source $HOME/.cargo/env"

# install mdbook and mdbook-linkcheck
cargo install mdbook
cargo install mdbook-linkcheck
cargo install mdbook-pdf
cargo install mdbook-toc
cargo install mdbook-mermaid

mdbook-mermaid install ${BBDIR}/docs/bb-note/

#  mdbook serve --open -p 3001
