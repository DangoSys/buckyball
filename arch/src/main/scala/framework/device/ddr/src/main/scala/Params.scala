package framework.device.ddr

import buckyball.config.DdrHardware
import scala.jdk.CollectionConverters._

import framework.ip.bus.chi
import framework.ip.bus.axi4
import upickle.default.{macroRW, ReadWriter}

case class Params(
  line:           chi.Params = chi.Params(),
  clients:        Int = 2,
  slotsPerClient: Int = 4,
  dataBits:       Int = 128,
  idBits:         Int = 4,
  idBase:         Int = 0,
  base:           BigInt = BigInt("80000000", 16),
  bytes:          BigInt = BigInt("20000000", 16),
  model:          BigInt = 0) {
  require(line.addressBits >= 6)
  require(clients > 0 && slotsPerClient > 0)
  require(Set(64, 128, 256, 512).contains(dataBits))
  val slots = clients * slotsPerClient
  require(slots > 1 && idBase >= 0 && BigInt(idBase) + slots <= (BigInt(1) << idBits))
  require(model <= bytes)
  val axi   = axi4.Params(line.addressBits, dataBits, idBits)
  val beats = 512 / dataBits
}

object Params {

  def fromProto(p: DdrHardware): Params =
    Params(
      chi.Params.fromProto(p.getLine),
      p.getClients,
      p.getSlotsPerClient,
      p.getDataBits,
      p.getIdBits,
      p.getIdBase,
      BigInt(java.lang.Long.toUnsignedString(p.getBase)),
      BigInt(java.lang.Long.toUnsignedString(p.getBytes)),
      BigInt(java.lang.Long.toUnsignedString(p.getModel))
    )

  implicit val chiRW: ReadWriter[chi.Params] = macroRW
  implicit val rw:    ReadWriter[Params]     = macroRW
}
