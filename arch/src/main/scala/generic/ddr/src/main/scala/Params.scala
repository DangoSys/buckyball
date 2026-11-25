package framework.generic.ddr

import buckyball.config.DdrHardware
import scala.jdk.CollectionConverters._

import framework.generic.bus.chi
import framework.generic.bus.axi4
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
  model:          BigInt = 0,
  beatRows:       Int) {
  require(line.addressBits >= 6)
  require(clients > 0 && slotsPerClient > 0)
  require(Set(64, 128, 256, 512, 1024).contains(dataBits * beatRows))
  val slots = clients * slotsPerClient
  require(slots > 1 && idBase >= 0 && BigInt(idBase) + slots <= (BigInt(1) << idBits))
  require(model <= bytes)
  // The system's AXI, a bank row a beat, and the DDR port, `beatRows` of them a beat.
  val axi   = axi4.Params(line.addressBits, dataBits, idBits)
  val port  = axi4.Params(line.addressBits, dataBits * beatRows, idBits)
  // A 64-byte line is `beats` port beats, or one of the `lanes` lines of a wider beat.
  val beats = math.max(1, 512 / port.dataBits)
  val lanes = math.max(1, port.dataBits / 512)
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
      BigInt(java.lang.Long.toUnsignedString(p.getModel)),
      p.getBeatRows
    )

  implicit val chiRW: ReadWriter[chi.Params] = macroRW
  implicit val rw:    ReadWriter[Params]     = macroRW
}
