package framework.core.memory.access.preflight.configs

import framework.core.memory.access.PhysicalRegion
import framework.core.memory.access.preflight.Params
import framework.ip.bus.chi
import upickle.default.{macroRW, ReadWriter}

case class PmpParams(addressBits: Int, entries: Int)
object PmpParams { implicit val rw: ReadWriter[PmpParams] = macroRW }

case class Config(permission: PmpParams, parameters: Params, regions: Seq[PhysicalRegion])

object Config {
  implicit val busRW:        ReadWriter[chi.Params]     = macroRW
  implicit val parametersRW: ReadWriter[Params]         = macroRW
  implicit val regionRW:     ReadWriter[PhysicalRegion] = macroRW
  implicit val rw:           ReadWriter[Config]         = macroRW
}
