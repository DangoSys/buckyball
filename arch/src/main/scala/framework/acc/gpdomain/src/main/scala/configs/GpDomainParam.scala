package framework.acc.gpdomain.configs

import buckyball.config.GpDomainConfig

import upickle.default._
import framework.acc.gpdomain.execution.configs.{Params => ExecutionParams}
import framework.acc.gpdomain.loader.configs.{Params => LoaderParams}
import framework.acc.gpdomain.memory.configs.{Params => MemoryParams}

case class GpDomainParam(
  enable:      Boolean,
  laneNumber:  Int,
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
    GpDomainParam(p.getEnable, p.getLaneNumber, p.getVLen, p.getELen, p.getIBufWords, p.getMemoryPorts)
  }

  implicit val rw: ReadWriter[GpDomainParam] = macroRW[GpDomainParam]

}
