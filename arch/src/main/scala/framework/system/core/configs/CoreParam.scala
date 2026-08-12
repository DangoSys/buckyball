package framework.system.core.configs

import upickle.default._

/**
 * Core Parameter
 */
case class CoreParam(
  coreDataBytes: Int,
  xLen:          Int,
  vaddrBits:     Int,
  paddrBits:     Int,
  pgIdxBits:     Int,
  nPMPs:         Int) // Physical Memory Protection entries, typically 8 or 16

object CoreParam {
  implicit val rw: ReadWriter[CoreParam] = macroRW

  def apply(): CoreParam = CoreParam(
    coreDataBytes = 0,
    xLen = 0,
    vaddrBits = 0,
    paddrBits = 0,
    pgIdxBits = 0,
    nPMPs = 0
  )

}
