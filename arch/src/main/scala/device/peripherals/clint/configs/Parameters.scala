package framework.device.peripherals.clint.configs

import buckyball.config.ClintHardware
import scala.jdk.CollectionConverters._

import upickle.default.{macroRW, ReadWriter}

case class ClintParams(
  hartIds:    Seq[Int],
  base:       BigInt = BigInt("2000000", 16),
  bytes:      BigInt = BigInt("10000", 16),
  tickCycles: Int = 1000) {
  require(tickCycles >= 1)
}

object ClintParams {

  def fromProto(p: ClintHardware): ClintParams = ClintParams(
    p.getHartIdsList.asScala.map(_.intValue).toSeq,
    BigInt(java.lang.Long.toUnsignedString(p.getBase)),
    BigInt(java.lang.Long.toUnsignedString(p.getBytes)),
    p.getTickCycles
  )

  implicit val rw: ReadWriter[ClintParams] = macroRW
}
