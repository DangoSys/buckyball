package framework.thread.acc.gpdomain.configs

import framework.thread.acc.gpdomain.execution.configs.{Params => ExecutionParams}
import framework.thread.acc.gpdomain.loader.configs.{Params => LoaderParams}
import framework.thread.acc.gpdomain.memory.configs.{Params => MemoryParams}
import buckyball.config.GpDomainConfig
import upickle.default._
import framework.thread.acc.gpdomain.configs.GpDomainParam

case class GpDomainParam(
  enable:      Boolean,
  fp64:        Boolean,
  laneNumber:  Int,
  baseInteger: Boolean,
  vLen:        Int,
  eLen:        Int,
  iBufWords:   Int,
  memoryPorts: Int) {
  val execution = ExecutionParams(laneNumber, vLen, eLen, iBufWords, memoryPorts)
  val memory    = MemoryParams(memoryPorts, eLen)
  val loader    = LoaderParams(iBufWords, memory.constBytes)
}

object GpDomainParam {

  def fromProto(p: GpDomainConfig): GpDomainParam = {
    require(p.hasEnable, "GP enable policy is required")
    require(p.hasFp64, "GP FP64 policy is required")
    GpDomainParam(
      p.getEnable,
      p.getFp64,
      p.getLaneNumber,
      p.getBaseInteger,
      p.getVLen,
      p.getELen,
      p.getIBufWords,
      p.getMemoryPorts
    )
  }

  implicit val rw: ReadWriter[GpDomainParam] = macroRW[GpDomainParam]

}
