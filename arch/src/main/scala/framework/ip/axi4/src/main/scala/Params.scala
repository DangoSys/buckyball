package framework.ip.bus.axi4

import upickle.default.{macroRW, ReadWriter}

import buckyball.config.AxiHardware

case class Params(addressBits: Int = 44, dataBits: Int = 128, idBits: Int = 4) {
  require(addressBits > 0 && idBits > 0)
  require(dataBits >= 8 && dataBits <= 1024 && (dataBits & (dataBits - 1)) == 0)
  val bytes = dataBits / 8
}

object Params {
  implicit val rw: ReadWriter[Params] = macroRW

  def fromProto(p: AxiHardware): Params = Params(p.getAddressBits, p.getDataBits, p.getIdBits)
}
