package examples.hart.xiangshan.configs

import framework.generic.contract.memory.access.mmu.configs.{Params => WalkerParam}

import framework.generic.acc.configs.AccParam
import framework.generic.acc.memdomain.tomem.configs.{Params => TransportParam}
import buckyball.config.HartPlacement
import buckyball.components.xiangshan.{Params => NativeParams}
import framework.generic.bus.{axi4, chi => chiBus}

import framework.generic.memory.interlock.{Params => TrackingParam}
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.acc.frontend.configs.FrontendParam
import framework.generic.acc.memdomain.configs.MemDomainParam
import framework.generic.acc.gpdomain.configs.GpDomainParam
import framework.generic.contract.configs.BallDomainParam

case class Params(
  chi:          chiBus.Params,
  axi:          axi4.Params,
  hartId:       Int,
  nodeId:       Int,
  homeId:       Int,
  resetVector:  BigInt,
  mtimeAddress: BigInt,
  permission:   PmpParams,
  tracking:     TrackingParam,
  walker:       WalkerParam,
  acc:          Option[AccParam]) {
  require(chi.nodeIdBits == 7 && chi.addressBits == 48 && chi.dataBits == 256)
  require(axi.addressBits == 48 && axi.dataBits == 128 && axi.idBits == 4)
  require(hartId >= 0 && hartId < 64 && nodeId > 0 && nodeId < homeId && homeId < 128)
  require((mtimeAddress & 7) == 0)
  require(permission.addressBits == 48 && permission.entries == 64)
  acc.foreach(a => require(!a.memDomain.sharedEnable && a.transport.axi == axi))
}

object Params {

  def fromProto(hart: HartPlacement): Params = {
    require(hart.getThreadIndicesCount == 0)
    val h      = hart
    val native = hart.getImplementation.unpack(classOf[NativeParams])
    val bus    = chiBus.Params.fromProto(h.getMemory.getHome.getRequesters)
    val acc    = Option.when(hart.hasAccelerator) {
      val a = hart.getAccelerator
      AccParam(
        FrontendParam.fromProto(a.getFrontend, hart.getParam.getXLen),
        MemDomainParam.fromProto(a.getMem, hart.getSharedMem, hart.getVirtualBankCount, 1),
        GpDomainParam.fromProto(a.getGpdomain),
        BallDomainParam.fromProto(a.getBalldomain),
        TransportParam.fromProto(a.getTransport)
      )
    }
    Params(
      bus,
      axi4.Params.fromProto(h.getMemory.getUncached.getAxi),
      h.getHartId,
      native.getNodeId,
      native.getHomeId,
      BigInt(java.lang.Long.toUnsignedString(h.getResetVector)),
      BigInt(java.lang.Long.toUnsignedString(h.getMtimeAddress)),
      PmpParams(48, 64),
      TrackingParam.fromProto(h.getMemory.getDcache.getTracking),
      WalkerParam(bus, h.getMemory.getAccess.getTranslationEntries),
      acc
    )
  }

}
