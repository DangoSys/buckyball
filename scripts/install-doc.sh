#!/bin/bash

set -e

# install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# install mdbook and mdbook-linkcheck
cargo install mdbook
cargo install mdbook-linkcheck
cargo install mdbook-pdf
