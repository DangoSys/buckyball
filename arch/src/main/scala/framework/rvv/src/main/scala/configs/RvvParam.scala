package framework.rvv.configs

import chisel3.util.isPow2
import upickle.default._

case class RvvParam(
  enable:      Boolean,
  laneNumber:  Int,
  vLen:        Int,
  eLen:        Int,
  iBufWords:   Int,
  memoryPorts: Int) {
  require(laneNumber > 0)
  require(vLen >= eLen && isPow2(vLen))
  require(eLen == 32 || eLen == 64)
  require(iBufWords > 0 && isPow2(iBufWords))
  require(memoryPorts > 0)
  require(laneNumber <= memoryPorts)

  val elementsPerRegister: Int = vLen / eLen
  val constBytes:          Int = 4096
  val stackBytes:          Int = 4096
}

object RvvParam {

  def apply(): RvvParam = RvvParam(
    enable = false,
    laneNumber = 4,
    vLen = 1024,
    eLen = 64,
    iBufWords = 1024,
    memoryPorts = 4
  )

  implicit val rw: ReadWriter[RvvParam] = macroRW[RvvParam]

}
