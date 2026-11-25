package framework.generic.contract.configs

import buckyball.config.{HartPlacement, ThreadInstance}
import framework.generic.bus.axi4
import framework.generic.contract.tlink.TLinkParam
import framework.generic.memory.spm

/**
 * What a hart knows of one of its threads: who it is, which program signature it runs and its
 * TLink. The thread's own parameters stay with the thread.
 */
case class ThreadSlot(
  id:        Int,
  name:      String,
  component: String,
  signature: BigInt,
  link:      TLinkParam)

object ThreadSlot {

  def fromProto(thread: ThreadInstance, hart: HartPlacement): ThreadSlot = {
    require(thread.hasLocalId && thread.hasSignature && thread.hasTransport)
    require(hart.hasTss && thread.hasCpu && thread.getCpu.getKind == "ant" && thread.getCpu.hasAnt)
    val tss = hart.getTss
    val ant = thread.getCpu.getAnt
    require(ant.hasTls, "Ant TLS configuration is required")
    val tls = ant.getTls
    require(tls.getDataBits == tss.getDataBits, "Ant TLS and TSS rows have one width")
    ThreadSlot(
      id = thread.getLocalId,
      name = thread.getRole,
      component = thread.getPkg,
      signature = BigInt(java.lang.Long.toUnsignedString(thread.getSignature)),
      link = TLinkParam(
        ant.getTaskBits,
        ant.getCodeBytes,
        BigInt(java.lang.Long.toUnsignedString(tls.getBase)),
        tls.getBytes,
        spm.Params(BigInt(java.lang.Long.toUnsignedString(tss.getBase)), tss.getBytes, tss.getDataBits),
        axi4.Params.fromProto(thread.getTransport.getAxi)
      )
    )
  }

}
