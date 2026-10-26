package framework.core.memory.configs

import framework.core.memory.access.configs.{Params => AccessParam}
import framework.core.memory.access.cache.configs.{Params => CacheParam}
import framework.core.memory.home.configs.{Params => HomeParam}
import framework.ip.bus.chi.rnf.RnfParams
import framework.core.memory.uncached.configs.UncachedParam
import buckyball.config.MemoryConfig
import scala.jdk.CollectionConverters._
import framework.core.memory.access.{AccessParams, PhysicalRegion}
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.ip.memory.coherence.configs.CoherenceParams

case class MemoryParam(
  memoryAccess: AccessParam,
  icache:       RnfParams,
  dcache:       CacheParam,
  home:         HomeParam,
  threadPorts:  Int,
  uncached:     UncachedParam)

object MemoryParam {

  def fromProto(p: MemoryConfig, threadPorts: Int): MemoryParam = {
    require(p.isInitialized, "Memory IP configuration is incomplete")
    val a         = p.getAccess
    val homep     = HomeParam(CoherenceParams.fromProto(p.getHome))
    val requester = homep.memory.chi
    MemoryParam(
      AccessParam(
        AccessParams(requester, a.getAccessTagBits),
        a.getRegionsList.asScala.map(PhysicalRegion.fromProto).toSeq,
        a.getTranslationEntries,
        a.getResultLineBits
      ),
      RnfParams.fromProto(p.getIcache, requester),
      CacheParam(RnfParams.fromProto(p.getDcache.getCache, requester), TrackingParam.fromProto(p.getDcache.getTracking)),
      homep,
      threadPorts,
      UncachedParam.fromProto(p.getUncached)
    )
  }

}
