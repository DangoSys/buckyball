# Custom configuration for BuckyBall FireSim builds
# This file overrides the default TARGET_CONFIG_PACKAGE to use our custom configs

# Only used in this projects makefrags
makefile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
makefile_dir := $(patsubst %/,%,$(dir $(makefile_path)))
chipyard_dir := $(abspath $(makefile_dir)/../../../../../../arch/thirdparty/chipyard)

# These point at the main class of the target's Chisel generator
DESIGN_PACKAGE ?= firechip.chip
DESIGN ?= FireSim

# Override to use our custom config package
TARGET_CONFIG_PACKAGE ?= sims.firesim
TARGET_CONFIG ?= FireSimBuckyballToyConfig

# These guide chisel elaboration of simulation components by MIDAS,
# including models and widgets.
PLATFORM_CONFIG_PACKAGE ?= firesim.firesim
PLATFORM_CONFIG ?= BaseF1Config

# Override project for the target.
TARGET_SBT_PROJECT := buckyball

# Point to our project directory
TARGET_SBT_DIR := $(abspath $(makefile_dir)/../../../../../../arch)
TARGET_SOURCE_DIRS := $(abspath $(makefile_dir)/../../../../../../arch/src/main/scala)

# SBT launcher
SBT ?= java -jar $(chipyard_dir)/scripts/sbt-launch.jar $(SBT_OPTS)
