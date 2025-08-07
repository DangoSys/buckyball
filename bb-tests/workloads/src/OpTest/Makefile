#!/bin/bash

CYDIR := $(shell git rev-parse --show-toplevel)

MLIR_BIN_DIR := ${CYDIR}/tools/buddy-mlir/build/bin

BUDDY_OPT := ${MLIR_BIN_DIR}/buddy-opt
BUDDY_TRANSLATE := ${MLIR_BIN_DIR}/buddy-translate
BUDDY_LLC := ${MLIR_BIN_DIR}/buddy-llc

bb_mvin_mvout-lower:
	@${BUDDY_OPT} ./bb_mvin_mvout.mlir -lower-buckyball -o log.mlir

bb_mvin_mvout-translate:
	@${BUDDY_OPT} ./bb_mvin_mvout.mlir -lower-buckyball | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir \
		-o log.ll

bb_mvin_mvout-run:
	@${BUDDY_OPT} ./bb_mvin_mvout.mlir -lower-buckyball | \
	${BUDDY_TRANSLATE} -buddy-to-llvmir | \
	${BUDDY_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+buddyext,+D -float-abi=hard \
		-relocation-model=pic \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -specs=htif_nano.specs -o a.out

