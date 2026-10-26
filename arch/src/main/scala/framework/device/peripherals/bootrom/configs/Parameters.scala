package framework.device.peripherals.bootrom.configs

import buckyball.config.BootRomHardware
import scala.jdk.CollectionConverters._

import upickle.default.{macroRW, ReadWriter}

case class BootRomParams(base: BigInt, bytes: BigInt, image: Seq[Byte]) {
  require(base % 64 == 0 && bytes % 64 == 0 && image.size <= bytes, "BootROM must be whole cache lines")
}

object BootRomParams {

  def fromProto(p: BootRomHardware): BootRomParams = BootRomParams(
    BigInt(java.lang.Long.toUnsignedString(p.getBase)),
    BigInt(java.lang.Long.toUnsignedString(p.getBytes)),
    p.getImageList.asScala.map(_.toByte).toSeq
  )

  implicit val rw: ReadWriter[BootRomParams] = macroRW
}
