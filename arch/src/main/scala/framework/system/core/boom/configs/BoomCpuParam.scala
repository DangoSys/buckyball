package framework.system.core.boom.configs

import upickle.default._
import freechips.rocketchip.rocket.{DCacheParams, ICacheParams}
import boom.v3.common.BoomCoreParams
import freechips.rocketchip.tile.FPUParams

case class BoomDCacheParam(
  nSets:  Int = 64,
  nWays:  Int = 4,
  nMSHRs: Int = 2)

object BoomDCacheParam {
  implicit val rw: ReadWriter[BoomDCacheParam] = macroRW
}

case class BoomICacheParam(
  nSets: Int = 64,
  nWays: Int = 4)

object BoomICacheParam {
  implicit val rw: ReadWriter[BoomICacheParam] = macroRW
}

case class BoomCpuParam(
  fetchWidth:    Int = 4,
  decodeWidth:   Int = 1,
  numRobEntries: Int = 32,
  dcache:        BoomDCacheParam = BoomDCacheParam(),
  icache:        BoomICacheParam = BoomICacheParam())

object BoomCpuParam {
  implicit val rw: ReadWriter[BoomCpuParam] = macroRW

  def toBoomCoreParams(p: BoomCpuParam): BoomCoreParams = BoomCoreParams(
    fetchWidth = p.fetchWidth,
    decodeWidth = p.decodeWidth,
    numRobEntries = p.numRobEntries,
    fpu = Some(FPUParams(sfmaLatency = 4, dfmaLatency = 4, divSqrt = true))
  )

  def toDCacheParams(p: BoomCpuParam, rowBits: Int): DCacheParams = DCacheParams(
    rowBits = rowBits,
    nSets = p.dcache.nSets,
    nWays = p.dcache.nWays,
    nMSHRs = p.dcache.nMSHRs,
    nTLBWays = 8
  )

  def toICacheParams(p: BoomCpuParam, rowBits: Int): ICacheParams = ICacheParams(
    rowBits = rowBits,
    nSets = p.icache.nSets,
    nWays = p.icache.nWays,
    fetchBytes = 2 * 4
  )

}
