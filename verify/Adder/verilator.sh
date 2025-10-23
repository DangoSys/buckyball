#!/bin/bash

BBDIR=$(git rev-parse --show-toplevel)

VERIFY_WORKSPACE=$BBDIR/verify
PICKERDIR=$BBDIR/thirdparty/picker

if ! command -v verible-verilog-syntax &> /dev/null
then
    echo "verible could not be found"
    echo "please add verible-verilog-syntax into path first"
    echo "https://chipsalliance.github.io/verible/verilog_syntax.html"
    echo "https://github.com/chipsalliance/verible/releases/tag/v0.0-3428-gcfcbb82b"
    exit
fi

rm -rf $VERIFY_WORKSPACE/out/
picker export Adder.v --autobuild false -w Adder.fst --sname Adder --tdir $VERIFY_WORKSPACE/out/Adder --sdir $PICKERDIR/template $@
# if python in $@, then it will generate python binding
if [[ $@ == *"python"* ]]; then
    cp $VERIFY_WORKSPACE/Adder/example.py $VERIFY_WORKSPACE/out/Adder/python/
else
    echo "unsupport"
fi

cd $VERIFY_WORKSPACE/out/Adder && make EXAMPLE=ON
