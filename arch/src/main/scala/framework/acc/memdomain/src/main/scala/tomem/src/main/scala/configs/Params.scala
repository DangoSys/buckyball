package framework.acc.memdomain.tomem.configs

import buckyball.config.TransportHardware
import scala.jdk.CollectionConverters._

import chisel3.util.log2Ceil
import framework.core.memory.access.PhysicalRegion
import framework.core.memory.access.preflight
import framework.ip.bus.{axi4, chi}
import framework.ip.memory.interlock
import upickle.default.{macroRW, ReadWriter}

case class Params(
  tracking: framework.ip.memory.interlock.Params,
  prepared: preflight.Params,
  axi:      axi4.Params,
  regions:  Seq[PhysicalRegion]) {
  val contexts  = 1 << log2Ceil(tracking.entries)
  val interlock = tracking.copy(entries = contexts)
  val mapping   = prepared.copy(contexts = contexts)
  require(tracking.idBits >= log2Ceil(contexts))
  require(prepared.idBits == tracking.idBits && prepared.maxRanges == tracking.maxRanges)
  require(tracking.addressBits == prepared.bus.addressBits && prepared.bus.addressBits == axi.addressBits)
  require(axi.dataBits == 128 && prepared.beatBytes == axi.bytes)
}

object Params {

  def fromProto(p: TransportHardware): Params = Params(
    interlock.Params.fromProto(p.getTracking),
    preflight.Params.fromProto(p.getPrepared),
    axi4.Params.fromProto(p.getAxi),
    p.getRegionsList.asScala.map(PhysicalRegion.fromProto).toSeq
  )

  implicit val chiRW:      ReadWriter[chi.Params]       = macroRW
  implicit val axiRW:      ReadWriter[axi4.Params]      = macroRW
  implicit val trackingRW: ReadWriter[interlock.Params] = macroRW
  implicit val preparedRW: ReadWriter[preflight.Params] = macroRW
  implicit val regionRW:   ReadWriter[PhysicalRegion]   = macroRW
  implicit val rw:         ReadWriter[Params]           = macroRW
}
