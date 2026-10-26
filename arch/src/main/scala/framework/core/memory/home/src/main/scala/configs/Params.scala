package framework.core.memory.home.configs

import framework.ip.memory.coherence.configs.CoherenceParams
import upickle.default.{macroRW, ReadWriter}

case class Params(memory: CoherenceParams)

object Params {
  implicit val chiRW:       ReadWriter[framework.ip.bus.chi.Params]                   = macroRW
  implicit val cacheRW:     ReadWriter[framework.ip.memory.cache.configs.CacheParams] = macroRW
  implicit val coherenceRW: ReadWriter[CoherenceParams]                               = macroRW
  implicit val rw:          ReadWriter[Params]                                        = macroRW
}
