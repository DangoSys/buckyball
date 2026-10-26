package framework.ip.memory.interlock

import buckyball.config.TrackingHardware

case class Params(
  entries:     Int = 4,
  addressBits: Int = 44,
  idBits:      Int = 8,
  lineBytes:   Int = 64,
  maxRanges:   Int = 8) {
  require(entries >= 2)
  require(addressBits >= 4 && idBits >= 1)
  require(lineBytes >= 8 && (lineBytes & (lineBytes - 1)) == 0)
  require(maxRanges >= 1 && (maxRanges & (maxRanges - 1)) == 0)
  val rangeIndexBits: Int = math.max(1, Integer.numberOfTrailingZeros(maxRanges))
  val rangeCountBits: Int = Integer.SIZE - Integer.numberOfLeadingZeros(maxRanges)
  val lineBits:       Int = Integer.numberOfTrailingZeros(lineBytes)
  require(addressBits > lineBits)
}

object Params {
  def fromProto(p: TrackingHardware): Params =
    Params(p.getEntries, p.getAddressBits, p.getIdBits, p.getLineBytes, p.getMaxRanges)
}
