#!/bin/bash

# exit script if any command fails
set -e
set -o pipefail

BBDIR=$(git rev-parse --show-toplevel)
T1DIR=${BBDIR}/arch/thirdparty/t1

source ${BBDIR}/scripts/utils.sh

cd ${BBDIR}
source ${BBDIR}/env.sh

if ! cargo --version > /dev/null 2>&1; then
  echo "cargo is not installed, this shoule be installed at the install-doc step"
  exit 1
fi

# To avoid the Buckyball project requiring root privileges,
# we opted for the slightly more cumbersome nix-user-chroot installation.
cargo install nix-user-chroot
mkdir -p -m 0755 ~/.nix
curl -L https://nixos.org/nix/install -o /tmp/nix-install.sh
nix-user-chroot ~/.nix bash /tmp/nix-install.sh

# if you see the following error
# error: filesystem error: read_symlink: Invalid argument [/home/xxxx/.nix-profile]
# this means you may repeated the execution of this Nix installation.


# enable nix-command and flakes
nix-user-chroot ~/.nix bash -c 'mkdir -p ~/.config/nix && echo "experimental-features = nix-command flakes" > ~/.config/nix/nix.conf'
replace_content ${BBDIR}/env.sh install-nix "if [ -z \"\$IN_NIX_ENV\" ] && ! command -v nix > /dev/null 2>&1; then
  IN_NIX_ENV=1 nix-user-chroot ~/.nix ${SHELL} -c \"source ${BBDIR}/env.sh && ${SHELL}\"
fi" "head"
