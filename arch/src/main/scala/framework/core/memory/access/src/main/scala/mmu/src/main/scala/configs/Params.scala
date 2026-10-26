package framework.core.memory.access.mmu.configs

import framework.ip.bus.chi
import upickle.default.{macroRW, ReadWriter}

case class Params(bus: chi.Params, entries: Int)

object Params {
  implicit val chiRW: ReadWriter[chi.Params] = macroRW
  implicit val rw:    ReadWriter[Params]     = macroRW
}
