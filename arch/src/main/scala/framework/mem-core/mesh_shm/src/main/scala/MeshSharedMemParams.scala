package memcore.memory.mesh_shm

import chisel3.util.log2Ceil
import memcore.memory.bank.BankSetParams

case class MeshCoreAttachment(bankIds: Seq[Int])

case class MeshSharedMemParams(
  rows:             Int,
  cols:             Int,
  entriesPerBank:   Int,
  dataBits:         Int,
  tagBits:          Int,
  cores:            Seq[MeshCoreAttachment],
  maxInFlight:      Int = 4,
  localBankBits:    Int = 10,
  visibleBanks:     Int = 0,
  externalChannels: Int = 0) {
  require(rows > 0 && cols > 0)
  require(entriesPerBank <= 65536)
  require(entriesPerBank >= 2 && (entriesPerBank & (entriesPerBank - 1)) == 0)
  require(dataBits > 0 && dataBits % 8 == 0)
  require(tagBits > 0 && localBankBits > 0 && cores.nonEmpty)
  require(cores.forall(c => c.bankIds.forall(id => id >= 0 && id < rows * cols)))

  require(externalChannels >= 0)
  require(maxInFlight > 0 && maxInFlight <= (1 << tagBits))
  val bankParams = BankSetParams(dataBits, 1, entriesPerBank, tagBits)
  val bankCount:        Int             = rows * cols
  val visibleBankCount: Int             = if (visibleBanks == 0) bankCount else visibleBanks
  require(visibleBankCount > 0 && visibleBankCount <= bankCount)
  require(cores.exists(_.bankIds.nonEmpty), "mesh requires a data endpoint")
  val totalChannels:    Int             = cores.map(_.bankIds.size).sum
  val channelLocations: Seq[(Int, Int)] =
    cores.flatMap(c => c.bankIds.map(id => (id / cols, id % cols)))
  val rowBits:          Int             = math.max(1, log2Ceil(rows))
  val colBits:          Int             = math.max(1, log2Ceil(cols))
  val bankBits:         Int             = math.max(1, log2Ceil(bankCount))
  val addressBits:      Int             = 16
  val channelBits:      Int             = math.max(1, log2Ceil(totalChannels + 1 + externalChannels))
  val coreBits:         Int             = math.max(1, log2Ceil(cores.size))

  val coreLocations: Seq[(Int, Int)] = cores.map { core =>
    if (core.bankIds.isEmpty) (0, 0) else (core.bankIds.head / cols, core.bankIds.head % cols)
  }

  val eventBits:      Int = rowBits + colBits + channelBits + addressBits + 2
  val packetUserBits: Int = eventBits + localBankBits + 9
  val maskBits:       Int = dataBits / 8

  // Core order in this sequence defines the flattened io.channels layout.
  def channelsForCore(coreIndex: Int): Range = {
    require(coreIndex >= 0 && coreIndex < cores.size)
    val start = cores.take(coreIndex).map(_.bankIds.size).sum
    start until (start + cores(coreIndex).bankIds.size)
  }

}

object MeshSharedMemParams {

  // Experimental 3x4 mesh: type 0/1/2 use 1/2/4 channels, with 2/2/1 cores.
  val prototype: MeshSharedMemParams = MeshSharedMemParams(
    rows = 3,
    cols = 4,
    entriesPerBank = 1024,
    dataBits = 128,
    tagBits = 8,
    cores = Seq(
      MeshCoreAttachment(Seq(0)),
      MeshCoreAttachment(Seq(3)),
      MeshCoreAttachment(Seq(4, 8)),
      MeshCoreAttachment(Seq(7, 11)),
      MeshCoreAttachment(Seq(5, 6, 9, 10))
    )
  )

}
