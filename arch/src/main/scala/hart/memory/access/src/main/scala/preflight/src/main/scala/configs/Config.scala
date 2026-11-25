package framework.hart.memory.access.preflight.configs

import framework.generic.contract.memory.access.PhysicalRegion
import framework.generic.contract.memory.access.preflight.Params

import framework.generic.bus.chi
import upickle.default.{macroRW, ReadWriter}
import framework.generic.contract.memory.access.preflight.configs.PmpParams

case class Config(permission: PmpParams, parameters: Params, regions: Seq[PhysicalRegion])

object Config {
  implicit val busRW:        ReadWriter[chi.Params]     = macroRW
  implicit val parametersRW: ReadWriter[Params]         = macroRW
  implicit val regionRW:     ReadWriter[PhysicalRegion] = macroRW
  implicit val rw:           ReadWriter[Config]         = macroRW
}
