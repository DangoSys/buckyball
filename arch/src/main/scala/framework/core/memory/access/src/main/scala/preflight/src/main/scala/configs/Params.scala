package framework.core.memory.access.preflight

import buckyball.config.PreparationHardware

import framework.core.memory.access.mmu.configs.{Params => WalkerParam}

import framework.ip.bus.chi

case class Params(
  bus:       chi.Params = chi.Params(),
  contexts:  Int = 4,
  maxRanges: Int = 8,
  idBits:    Int = 8,
  beatBytes: Int = 16) {
  val walker = WalkerParam(bus, 0)

  require(contexts >= 2)
  require(maxRanges == 8)
  require(idBits > 0 && bus.addressBits >= 12 && bus.addressBits < 64)
  require(beatBytes >= 8 && beatBytes <= 64 && (beatBytes & (beatBytes - 1)) == 0)
}

object Params {

  def fromProto(p: PreparationHardware): Params = Params(
    chi.Params.fromProto(p.getBus),
    p.getContexts,
    p.getMaxRanges,
    p.getIdBits,
    p.getBeatBytes
  )

}
