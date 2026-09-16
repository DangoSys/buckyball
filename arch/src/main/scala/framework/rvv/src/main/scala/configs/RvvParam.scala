package framework.rvv.configs

import chisel3.util.isPow2

case class RvvParam(
  laneNumber:  Int,
  vLen:        Int,
  eLen:        Int,
  iBufWords:   Int,
  dBufWords:   Int,
  memoryPorts: Int) {
  require(laneNumber > 0)
  require(vLen >= eLen && isPow2(vLen))
  require(eLen == 32)
  require(iBufWords > 0 && isPow2(iBufWords))
  require(dBufWords > 0 && isPow2(dBufWords))
  require(memoryPorts > 0)
  require(laneNumber <= memoryPorts)

  val elementsPerRegister: Int = vLen / eLen
}

object RvvParam {

  def apply(): RvvParam = RvvParam(
    laneNumber = 1,
    vLen = 256,
    eLen = 32,
    iBufWords = 1024,
    dBufWords = 8192,
    memoryPorts = 1
  )

}
