#!/bin/bash

BBDIR=$(git rev-parse --show-toplevel)
source ${BBDIR}/scripts/utils.sh

if [ -z "$CONDA_PREFIX" ]; then
    echo "CONDA environment is not set, please source the env.sh"
    exit 1
else
    PREFIX=$CONDA_PREFIX/picker
    echo "Picker will be installed to $PREFIX"
fi

# ==================================================================
# install picker
# ==================================================================

# === install dependencies for picker
mkdir -p $BBDIR/tmp && cd $BBDIR/tmp
wget "https://github.com/chipsalliance/verible/releases/download/v0.0-4007-g98bdb38a/verible-v0.0-4007-g98bdb38a-linux-static-x86_64.tar.gz"
tar -xzf verible-v0.0-4007-g98bdb38a-linux-static-x86_64.tar.gz
mv verible-v0.0-4007-g98bdb38a/bin/* $PREFIX
cd $BBDIR && rm -rf $BBDIR/tmp

conda install swig=4.2.0 -y

# === install picker
cd $BBDIR/thirdparty/picker
make init

cd $BBDIR/thirdparty/picker
make -j$(nproc) ARGS="-DCMAKE_INSTALL_PREFIX=$PREFIX"
sudo -E make install

replace_content ${BBDIR}/env.sh picker-install "\
export PATH=$PREFIX/bin:\$PATH"
