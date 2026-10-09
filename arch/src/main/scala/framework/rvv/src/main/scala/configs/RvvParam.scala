package framework.rvv.configs

import chisel3.util.{isPow2, log2Ceil}
import upickle.default._

case class RvvParam(
  enable:      Boolean,
  laneNumber:  Int,
  vLen:        Int,
  eLen:        Int,
  iBufWords:   Int,
  memoryPorts: Int) {
  require(laneNumber > 0)
  require(vLen >= 128 && vLen >= eLen && isPow2(vLen))
  require(eLen == 32 || eLen == 64)
  require(iBufWords > 0 && isPow2(iBufWords))
  require(memoryPorts > 0)
  require(laneNumber <= memoryPorts)

  val wordBits:         Int = eLen
  val wordOffsetBits:   Int = log2Ceil(wordBits)
  val maxSew:           Int = log2Ceil(eLen / 8)
  val wordsPerRegister: Int = vLen / wordBits
  val registerBanks:    Int = math.min(laneNumber, wordsPerRegister)
  require(isPow2(registerBanks), "RVV register bank count must be a power of two")

  val elementsPerRegister: Int = vLen / eLen
  val constBytes:          Int = 4096
  val stackBytes:          Int = 4096
}

object RvvParam {

  def apply(): RvvParam = RvvParam(
    enable = false,
    laneNumber = 4,
    vLen = 1024,
    eLen = 32,
    iBufWords = 1024,
    memoryPorts = 4
  )

  implicit val rw: ReadWriter[RvvParam] = macroRW[RvvParam]

}
