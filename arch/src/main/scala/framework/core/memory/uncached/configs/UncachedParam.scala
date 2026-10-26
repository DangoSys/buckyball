package framework.core.memory.uncached.configs

import buckyball.config.UncachedConfig
import framework.ip.bus.axi4
import upickle.default.{macroRW, ReadWriter}

case class UncachedParam(ports: Int, axi: axi4.Params) {
  require(ports > 0)
}

object UncachedParam {
  def fromProto(p: UncachedConfig): UncachedParam =
    UncachedParam(p.getPorts, axi4.Params.fromProto(p.getAxi))
  implicit val rw: ReadWriter[UncachedParam] = macroRW
}
