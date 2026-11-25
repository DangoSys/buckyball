package framework.thread.configs

import buckyball.config.{Design, HartPlacement, ThreadInstance}
import framework.generic.contract.configs.{ThreadSlot}
import framework.thread.acc.configs.{AccParam}
import framework.generic.contract.tlink.TLinkParam
import scala.jdk.CollectionConverters._

case class ThreadParam(
  id:          Int,
  name:        String,
  component:   String,
  rtlClass:    String,
  commandBits: Int,
  acc:         Option[AccParam],
  link:        TLinkParam) {
  require(commandBits == 64, "Ant implements a 64-bit RoCC command interface")
}

object ThreadParam {

  def fromProto(thread: ThreadInstance, hart: HartPlacement): ThreadParam = {
    val slot          = ThreadSlot.fromProto(thread, hart)
    val accelerator   = AccParam.fromProto(thread, hart)
    val privateMemory = accelerator.memDomain.copy(
      sharedPorts = 0,
      sharedEnable = false,
      sharedEntries = 0,
      sharedBankEntries = 0,
      sharedBankNum = 0,
      sharedInputChannels = 0,
      sharedDefaultGroupCount = 0
    )
    require(accelerator.transport.axi == slot.link.dma)
    ThreadParam(
      id = slot.id,
      name = slot.name,
      component = slot.component,
      rtlClass = thread.getRtlClass,
      commandBits = hart.getParam.getXLen,
      acc = Some(accelerator.copy(
        memDomain = privateMemory,
        frontend = accelerator.frontend.copy(shared_vbank_base = privateMemory.virtualBankCount)
      )),
      link = slot.link
    )
  }

  /** The threads of `hart`, in the order of its slots. */
  def ofHart(design: Design, hart: HartPlacement): Seq[ThreadParam] =
    hart.getThreadIndicesList.asScala.map { index =>
      val thread = design.getThreads(index.intValue)
      require(thread.getIndex == index.intValue)
      fromProto(thread, hart)
    }.toSeq

}
