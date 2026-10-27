package examples.core.boom.configs

import buckyball.config.CorePlacement
import framework.ip.bus.{axi4, chi => chiBus}

import framework.core.memory.access.preflight.configs.PmpParams
import framework.ip.memory.interlock.{Params => TrackingParam}
import framework.core.memory.access.mmu.configs.{Params => WalkerParam}
import framework.acc.configs.AccParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.memdomain.tomem.configs.{Params => TransportParam}

case class Params(
  chi:          chiBus.Params,
  axi:          axi4.Params,
  hartId:       Int,
  resetVector:  BigInt,
  mtimeAddress: BigInt,
  permission:   PmpParams,
  tracking:     TrackingParam,
  walker:       WalkerParam,
  acc:          Option[AccParam]) {
  require(chi.nodeIdBits == 7 && chi.addressBits == 44 && chi.dataBits == 256)
  require(axi.addressBits == 44 && axi.dataBits == 128 && axi.idBits == 4)
  require(hartId == 0)
  require((mtimeAddress & 7) == 0)
  require(permission.addressBits == 44 && permission.entries == 8)
  acc.foreach(a => require(!a.memDomain.sharedEnable && a.transport.axi == axi))
}

object Params {

  def fromProto(core: CorePlacement): Params = {
    require(core.getThreadIndicesCount == 0)
    require(core.getImplementation.unpack(classOf[buckyball.components.boom.Params]).getProfile == "SmallBoomV4")
    val h   = core
    val bus = chiBus.Params.fromProto(h.getMemory.getHome.getRequesters)
    val acc = Option.when(core.hasAccelerator) {
      val a = core.getAccelerator
      AccParam(
        FrontendParam.fromProto(a.getFrontend, core.getParam.getXLen, core.getHartId),
        MemDomainParam.fromProto(a.getMem, core.getSharedMem, core.getVirtualBankCount, 1),
        GpDomainParam.fromProto(a.getGpdomain),
        BallDomainParam.fromProto(a.getBalldomain),
        TransportParam.fromProto(a.getTransport)
      )
    }
    Params(
      bus,
      axi4.Params.fromProto(h.getMemory.getUncached.getAxi),
      h.getHartId,
      BigInt(java.lang.Long.toUnsignedString(h.getResetVector)),
      BigInt(java.lang.Long.toUnsignedString(h.getMtimeAddress)),
      PmpParams(44, 8),
      TrackingParam.fromProto(h.getMemory.getDcache.getTracking),
      WalkerParam(bus, h.getMemory.getAccess.getTranslationEntries),
      acc
    )
  }

}
