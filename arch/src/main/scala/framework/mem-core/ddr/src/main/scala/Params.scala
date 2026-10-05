package memcore.memory.ddr

import memcore.bus.chi
import memcore.bus.axi4

case class Params(
  line:           chi.Params = chi.Params(),
  clients:        Int = 2,
  slotsPerClient: Int = 4,
  dataBits:       Int = 128,
  idBits:         Int = 4,
  idBase:         Int = 0) {
  require(line.addressBits >= 6)
  require(clients > 0 && slotsPerClient > 0)
  require(Set(64, 128, 256, 512).contains(dataBits))
  val slots = clients * slotsPerClient
  require(slots > 1 && idBase >= 0 && BigInt(idBase) + slots <= (BigInt(1) << idBits))
  val axi   = axi4.Params(line.addressBits, dataBits, idBits)
  val beats = 512 / dataBits
}
