package memcore.memory.preflight

import memcore.bus.chi

case class Params(
  bus:       chi.Params = chi.Params(),
  contexts:  Int = 4,
  maxRanges: Int = 8,
  idBits:    Int = 8,
  beatBytes: Int = 16) {
  require(contexts >= 2 && (contexts & (contexts - 1)) == 0)
  require(maxRanges == 8)
  require(idBits > 0 && bus.addressBits >= 12 && bus.addressBits < 64)
  require(beatBytes >= 8 && beatBytes <= 64 && (beatBytes & (beatBytes - 1)) == 0)
}
