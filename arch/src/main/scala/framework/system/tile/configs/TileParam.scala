package framework.system.tile.configs

import upickle.default._

case class TileParam(
  coreDataBytes: Int,
  xLen:          Int,
  vaddrBits:     Int,
  paddrBits:     Int,
  pgIdxBits:     Int,
  pgLevels:      Int,
  nPMPs:         Int) // Physical Memory Protection entries, typically 8 or 16

object TileParam {
  implicit val rw: ReadWriter[TileParam] = macroRW

  def apply(): TileParam = TileParam(
    coreDataBytes = 0,
    xLen = 0,
    vaddrBits = 0,
    paddrBits = 0,
    pgIdxBits = 0,
    pgLevels = 0,
    nPMPs = 0
  )

}
