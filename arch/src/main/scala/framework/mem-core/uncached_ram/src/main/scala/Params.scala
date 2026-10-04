package memcore.memory.uncached_ram

import memcore.bus.chi.{Params => ChiParams}

case class Params(
  line:          ChiParams = ChiParams(),
  cpuHartIds:    Seq[BigInt] = Seq(BigInt(0), BigInt(1)),
  lineAgents:    Int = 2,
  slots:         Int = 8,
  tagBits:       Int = 6,
  base:          BigInt = BigInt("80000000", 16),
  bytes:         BigInt = BigInt(16) << 20,
  externalPorts: Int = 0) {
  require(cpuHartIds.nonEmpty && cpuHartIds.distinct.size == cpuHartIds.size && cpuHartIds.forall(_ >= 0))
  require(externalPorts >= 0)
  require(lineAgents > 0 && slots >= 4 && (slots & (slots - 1)) == 0)
  require(tagBits > 0 && line.addressBits < 64 && BigInt(slots) <= (BigInt(1) << line.txnIdBits))
  require(base >= 0 && bytes > 0 && base % 64 == 0 && bytes % 64 == 0 && base + bytes <= (BigInt(1) << line.addressBits))
  val cpus         = cpuHartIds.size
  val sources      = cpus + lineAgents
  val ports        = 2
  val slotsPerPort = slots / ports
}
