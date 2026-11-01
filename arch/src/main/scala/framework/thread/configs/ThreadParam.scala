package framework.thread.configs

import buckyball.config.{CorePlacement, ThreadInstance}
import framework.ip.memory.spm
import framework.acc.configs.AccParam
import framework.thread.ant.configs.AntParam

case class ThreadParam(
  id:          Int,
  name:        String,
  component:   String,
  rtlClass:    String,
  signature:   BigInt,
  commandBits: Int,
  acc:         Option[AccParam],
  ant:         Option[AntParam]) {
  require(ant.isEmpty || commandBits == 64, "Ant implements a 64-bit RoCC command interface")
}

object ThreadParam {

  def fromProto(thread: ThreadInstance, core: CorePlacement): ThreadParam = {
    require(thread.hasLocalId && thread.hasSignature)
    require(core.hasTss && thread.hasCpu && thread.getCpu.getKind == "ant" && thread.getCpu.hasAnt)
    val tss           = core.getTss
    val shared        = spm.Params(BigInt(java.lang.Long.toUnsignedString(tss.getBase)), tss.getBytes, tss.getDataBits)
    val accelerator   = AccParam.fromProto(thread, core)
    val privateMemory = accelerator.memDomain.copy(
      sharedPorts = 0,
      sharedEnable = false,
      sharedEntries = 0,
      sharedBankEntries = 0,
      sharedBankNum = 0,
      sharedInputChannels = 0,
      sharedDefaultGroupCount = 0
    )
    ThreadParam(
      id = thread.getLocalId,
      name = thread.getRole,
      component = thread.getPkg,
      rtlClass = thread.getRtlClass,
      signature = BigInt(java.lang.Long.toUnsignedString(thread.getSignature)),
      commandBits = core.getParam.getXLen,
      acc = Some(accelerator.copy(
        memDomain = privateMemory,
        frontend = accelerator.frontend.copy(shared_vbank_base = privateMemory.virtualBankCount)
      )),
      ant = Some(AntParam.fromProto(thread.getCpu.getAnt, shared))
    )
  }

}
