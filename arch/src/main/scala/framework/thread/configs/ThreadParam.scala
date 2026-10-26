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
    val tss    = core.getTss
    val shared = spm.Params(BigInt(java.lang.Long.toUnsignedString(tss.getBase)), tss.getBytes, tss.getDataBits)
    ThreadParam(
      id = thread.getLocalId,
      name = thread.getRole,
      component = thread.getPkg,
      rtlClass = thread.getRtlClass,
      signature = BigInt(java.lang.Long.toUnsignedString(thread.getSignature)),
      commandBits = core.getParam.getXLen,
      acc = Some(AccParam.fromProto(thread, core)),
      ant = Some(AntParam.fromProto(thread.getCpu.getAnt, shared))
    )
  }

}
