package framework.system.core.boom.configs

import upickle.default._

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

/** BOOM configuration from the chip PB. The explicit Tile does not instantiate BOOM yet. */
case class BoomCpuParam(
  fetchWidth:    Int = 4,
  decodeWidth:   Int = 1,
  numRobEntries: Int = 32,
  dcache:        BoomDCacheParam = BoomDCacheParam(),
  icache:        BoomICacheParam = BoomICacheParam())

object BoomCpuParam {
  implicit val rw: ReadWriter[BoomCpuParam] = macroRW
}
