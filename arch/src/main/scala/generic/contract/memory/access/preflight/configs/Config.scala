package framework.generic.contract.memory.access.preflight.configs

import framework.generic.contract.memory.access.PhysicalRegion
import framework.generic.contract.memory.access.preflight.Params

import framework.generic.bus.chi
import upickle.default.{macroRW, ReadWriter}
import framework.generic.contract.memory.access.preflight.configs.PmpParams

case class PmpParams(addressBits: Int, entries: Int)

object PmpParams { implicit val rw: ReadWriter[PmpParams] = macroRW }
