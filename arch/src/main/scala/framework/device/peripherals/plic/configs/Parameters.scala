package framework.device.peripherals.plic.configs

import buckyball.config.PlicHardware
import scala.jdk.CollectionConverters._

import upickle.default.{macroRW, ReadWriter}

case class PlicParams(
  hartIds:      Seq[Int],
  base:         BigInt = BigInt("c000000", 16),
  bytes:        BigInt = BigInt("4000000", 16),
  sources:      Int = 1,
  priorityBits: Int = 3) {
  require(sources >= 1 && sources <= 31 && priorityBits >= 1 && priorityBits <= 32)
}

object PlicParams {

  def fromProto(p: PlicHardware): PlicParams = PlicParams(
    p.getHartIdsList.asScala.map(_.intValue).toSeq,
    BigInt(java.lang.Long.toUnsignedString(p.getBase)),
    BigInt(java.lang.Long.toUnsignedString(p.getBytes)),
    p.getSources,
    p.getPriorityBits
  )

  implicit val rw: ReadWriter[PlicParams] = macroRW
}
