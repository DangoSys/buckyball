package framework.thread.acc.memdomain.tomem.configs

import framework.generic.contract.memory.access.PhysicalRegion

import framework.thread.acc.memdomain.tomem.configs.Params

import buckyball.config.TransportHardware
import scala.jdk.CollectionConverters._
import chisel3.util.log2Ceil
import framework.generic.bus.{axi4, chi}
import framework.generic.memory.interlock
import upickle.default.{macroRW, ReadWriter}

case class Params(
  tracking: framework.generic.memory.interlock.Params,
  prepared: framework.generic.contract.memory.access.preflight.Params,
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
    framework.generic.contract.memory.access.preflight.Params.fromProto(p.getPrepared),
    axi4.Params.fromProto(p.getAxi),
    p.getRegionsList.asScala.map(PhysicalRegion.fromProto).toSeq
  )

  implicit val chiRW:      ReadWriter[chi.Params]                                                = macroRW
  implicit val axiRW:      ReadWriter[axi4.Params]                                               = macroRW
  implicit val trackingRW: ReadWriter[interlock.Params]                                          = macroRW
  implicit val preparedRW: ReadWriter[framework.generic.contract.memory.access.preflight.Params] = macroRW
  implicit val regionRW:   ReadWriter[PhysicalRegion]                                            = macroRW
  implicit val rw:         ReadWriter[Params]                                                    = macroRW
}
