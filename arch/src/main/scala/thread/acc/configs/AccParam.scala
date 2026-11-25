package framework.thread.acc.configs

import framework.thread.acc.memdomain.tomem.configs.{Params => TransportParam}
import framework.thread.acc.configs.AccParam

import buckyball.config.{HartPlacement, ThreadInstance}
import upickle.default.{macroRW, ReadWriter}
import framework.generic.contract.configs.BallDomainParam
import framework.thread.acc.frontend.configs.FrontendParam
import framework.thread.acc.gpdomain.configs.GpDomainParam
import framework.thread.acc.memdomain.configs.MemDomainParam

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
    frontend.max_private_vbank < memDomain.virtualBankCount,
    "Private bank upper bound must be below virtualBankCount"
  )
  if (memDomain.sharedEnable) {
    require(
      frontend.shared_vbank_base > frontend.max_private_vbank,
      "Shared bank base must be above the private bank range"
    )
    require(
      frontend.shared_vbank_base <= memDomain.virtualBankCount,
      "Shared bank base must not exceed virtualBankCount"
    )
  }
}

object AccParam {

  def fromProto(hart: HartPlacement): AccParam = {
    require(hart.hasAccelerator, "Hart has no local accelerator")
    val acc = hart.getAccelerator
    AccParam(
      FrontendParam.fromProto(acc.getFrontend, hart.getParam.getXLen),
      MemDomainParam.fromProto(acc.getMem, hart.getSharedMem, hart.getVirtualBankCount, 1),
      GpDomainParam.fromProto(acc.getGpdomain),
      BallDomainParam.fromProto(acc.getBalldomain),
      TransportParam.fromProto(acc.getTransport)
    )
  }

  def fromProto(thread: ThreadInstance, hart: HartPlacement): AccParam = {
    require(thread.hasTransport)
    AccParam(
      FrontendParam.fromProto(thread.getFrontend, hart.getParam.getXLen),
      MemDomainParam.fromProto(thread.getMem, hart.getSharedMem, hart.getVirtualBankCount, hart.getThreadIndicesCount),
      GpDomainParam.fromProto(thread.getGpdomain),
      BallDomainParam.fromProto(thread.getBalldomain),
      TransportParam.fromProto(thread.getTransport)
    )
  }

  implicit val rw: ReadWriter[AccParam] = macroRW
}
