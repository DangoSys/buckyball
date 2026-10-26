package framework.core.memory.access.cache.configs

import framework.ip.bus.chi.rnf.RnfParams
import framework.ip.memory.interlock.{Params => TrackingParams}
import upickle.default.{macroRW, ReadWriter}

case class Params(cache: RnfParams, tracking: TrackingParams)

object Params {
  implicit val chiRW:      ReadWriter[framework.ip.bus.chi.Params]                   = macroRW
  implicit val cacheRW:    ReadWriter[framework.ip.memory.cache.configs.CacheParams] = macroRW
  implicit val rnfRW:      ReadWriter[RnfParams]                                     = macroRW
  implicit val trackingRW: ReadWriter[TrackingParams]                                = macroRW
  implicit val rw:         ReadWriter[Params]                                        = macroRW
}
