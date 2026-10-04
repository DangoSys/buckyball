package sims.verilator

import sims.soc.SystemTarget

/** Pebble on the explicit System: one compute core with private banks and no task controller. */
class PebbleVerilatorTarget extends SystemTarget("../examples/chips/pebble/configs/generated/chip.pb")
