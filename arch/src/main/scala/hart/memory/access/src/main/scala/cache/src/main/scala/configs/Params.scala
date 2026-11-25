package framework.hart.memory.access.cache.configs

import framework.generic.bus.chi.rnf.RnfParams
import framework.generic.memory.interlock.{Params => TrackingParams}
import upickle.default.{macroRW, ReadWriter}

case class Params(cache: RnfParams, tracking: TrackingParams)

object Params {
  implicit val chiRW:      ReadWriter[framework.generic.bus.chi.Params]                   = macroRW
  implicit val cacheRW:    ReadWriter[framework.generic.memory.cache.configs.CacheParams] = macroRW
  implicit val rnfRW:      ReadWriter[RnfParams]                                          = macroRW
  implicit val trackingRW: ReadWriter[TrackingParams]                                     = macroRW
  implicit val rw:         ReadWriter[Params]                                             = macroRW
}
