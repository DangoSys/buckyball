package sims.verilator

import sims.soc.SystemTarget

/** Goban on the explicit System: Linux scheduler core 0 and four compute cores. */
class GobanVerilatorTarget extends SystemTarget("../examples/chips/goban/configs/generated/chip.pb")
