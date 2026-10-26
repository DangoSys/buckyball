package framework.acc.configs

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.gpdomain.configs.GpDomainParam
import framework.acc.memdomain.configs.MemDomainParam
import buckyball.config.{CorePlacement, ThreadInstance}
import framework.acc.memdomain.tomem.configs.{Params => TransportParam}
import upickle.default.{macroRW, ReadWriter}

case class AccParam(
  frontend:   FrontendParam,
  memDomain:  MemDomainParam,
  gpDomain:   GpDomainParam,
  ballDomain: BallDomainParam,
  transport:  TransportParam) {
  require(
    BigInt(memDomain.virtualBankCount) <= (BigInt(1) << frontend.bank_id_len),
    "virtualBankCount must fit the frontend bank ID width"
  )
  require(
    frontend.vbank_id_upper_bound < memDomain.virtualBankCount,
    "Private bank upper bound must be below virtualBankCount"
  )
  if (memDomain.sharedEnable) {
    require(
      frontend.shared_bank_id_base > frontend.vbank_id_upper_bound,
      "Shared bank base must be above the private bank range"
    )
    require(
      frontend.shared_bank_id_base <= memDomain.virtualBankCount,
      "Shared bank base must not exceed virtualBankCount"
    )
  }
}

object AccParam {

  def fromProto(core: CorePlacement): AccParam = {
    require(core.hasAccelerator, "Core has no local accelerator")
    val acc = core.getAccelerator
    AccParam(
      FrontendParam.fromProto(acc.getFrontend, core.getParam.getXLen, core.getHartId),
      MemDomainParam.fromProto(acc.getMem, core.getSharedMem, core.getVirtualBankCount, 1),
      GpDomainParam.fromProto(acc.getGpdomain),
      BallDomainParam.fromProto(acc.getBalldomain),
      TransportParam.fromProto(acc.getTransport)
    )
  }

  def fromProto(thread: ThreadInstance, core: CorePlacement): AccParam = {
    require(thread.hasTransport)
    AccParam(
      FrontendParam.fromProto(thread.getFrontend, core.getParam.getXLen, thread.getLocalId),
      MemDomainParam.fromProto(thread.getMem, core.getSharedMem, core.getVirtualBankCount, core.getThreadIndicesCount),
      GpDomainParam.fromProto(thread.getGpdomain),
      BallDomainParam.fromProto(thread.getBalldomain),
      TransportParam.fromProto(thread.getTransport)
    )
  }

  implicit val rw: ReadWriter[AccParam] = macroRW
}
