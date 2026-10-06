package sims.verilator

import sims.soc.SystemTarget

/** Poly on the explicit System: a CPU main tile and eight homogeneous compute tiles. */
class PolyVerilatorTarget extends SystemTarget("../examples/chips/poly/configs/generated/chip.pb")
