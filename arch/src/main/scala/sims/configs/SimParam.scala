package sims.configs

import buckyball.config.SimHardware

import upickle.default._

case class MemoryParam(
  lineBytes: Int,
  clockHz:   BigInt)

object MemoryParam { implicit val rw: ReadWriter[MemoryParam] = macroRW }

/** Differential debugging may delay completion; passive trace does not use this flag. */
case class SimParam(diffTest: Boolean, memory: MemoryParam)

object SimParam {

  def fromProto(p: SimHardware): SimParam = {
    require(p.isInitialized, "Simulation configuration is incomplete")
    val memory = p.getMemory
    SimParam(
      p.getDiffTest,
      MemoryParam(
        memory.getLineBytes,
        BigInt(java.lang.Long.toUnsignedString(memory.getClockHz))
      )
    )
  }

  implicit val rw: ReadWriter[SimParam] = macroRW
}
