package framework.hart.memory.home.configs

import framework.generic.memory.coherence.configs.CoherenceParams
import upickle.default.{macroRW, ReadWriter}

case class Params(memory: CoherenceParams)

object Params {
  implicit val chiRW:       ReadWriter[framework.generic.bus.chi.Params]                   = macroRW
  implicit val cacheRW:     ReadWriter[framework.generic.memory.cache.configs.CacheParams] = macroRW
  implicit val coherenceRW: ReadWriter[CoherenceParams]                                    = macroRW
  implicit val rw:          ReadWriter[Params]                                             = macroRW
}
