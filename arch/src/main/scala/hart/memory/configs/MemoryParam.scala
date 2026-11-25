package framework.hart.memory.configs

import framework.generic.contract.memory.access.AccessParams

import framework.generic.contract.memory.access.PhysicalRegion

import framework.hart.memory.access.configs.{Params => AccessParam}
import framework.hart.memory.access.cache.configs.{Params => CacheParam}
import framework.hart.memory.home.configs.{Params => HomeParam}
import framework.generic.bus.chi.rnf.RnfParams
import framework.hart.memory.uncached.configs.UncachedParam
import buckyball.config.MemoryConfig
import scala.jdk.CollectionConverters._
import framework.generic.memory.interlock.{Params => TrackingParam}
import framework.generic.memory.coherence.configs.CoherenceParams

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
    require(
      a.getResultLineBits != 512 || p.getIcache.getLineResult,
      "CPU fetch line response requires full-line instruction cache results"
    )
    MemoryParam(
      AccessParam(
        AccessParams(requester, a.getTagBits),
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
