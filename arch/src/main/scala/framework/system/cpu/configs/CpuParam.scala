package framework.system.cpu.configs

import upickle.default._

case class CpuParam(
  coreDataBytes: Int,
  xLen:          Int,
  vaddrBits:     Int,
  paddrBits:     Int,
  pgIdxBits:     Int,
  pgLevels:      Int,
  nPMPs:         Int) // Physical Memory Protection entries, typically 8 or 16

object CpuParam {
  implicit val rw: ReadWriter[CpuParam] = macroRW

  def apply(): CpuParam = CpuParam(
    coreDataBytes = 0,
    xLen = 0,
    vaddrBits = 0,
    paddrBits = 0,
    pgIdxBits = 0,
    pgLevels = 0,
    nPMPs = 0
  )

}
