package memcore.memory.mesh_shm

import chisel3.util.log2Ceil
import framework.top.GlobalConfig
import framework.memdomain.configs.MemDomainParam

case class MeshCoreAttachment(bankIds: Seq[Int])

case class MeshSharedMemParams(
  global:        GlobalConfig,
  rows:          Int,
  cols:          Int,
  tagBits:       Int,
  cores:         Seq[MeshCoreAttachment],
  localBankBits: Int = 10,
  stagingBankId: Int = 0,
  visibleBanks:  Int = 0) {
  val entriesPerBank: Int = global.memDomain.bankEntries
  val dataBits:       Int = global.memDomain.bankWidth
  require(rows > 0 && cols > 0)
  require(entriesPerBank >= 2 && (entriesPerBank & (entriesPerBank - 1)) == 0)
  require(dataBits > 0 && dataBits % 8 == 0)
  require(global.memDomain.bankMaskLen == dataBits / 8)
  require(tagBits > 0 && localBankBits > 0 && cores.nonEmpty)
  require(cores.forall(c => c.bankIds.nonEmpty && c.bankIds.forall(id => id >= 0 && id < rows * cols)))

  val bankCount:        Int             = rows * cols
  val visibleBankCount: Int             = if (visibleBanks == 0) bankCount else visibleBanks
  require(visibleBankCount > 0 && visibleBankCount <= bankCount)
  require(stagingBankId >= 0 && stagingBankId < bankCount)
  val totalChannels:    Int             = cores.map(_.bankIds.size).sum
  val channelLocations: Seq[(Int, Int)] =
    cores.flatMap(c => c.bankIds.map(id => (id / cols, id % cols)))
  val rowBits:          Int             = math.max(1, log2Ceil(rows))
  val colBits:          Int             = math.max(1, log2Ceil(cols))
  val bankBits:         Int             = math.max(1, log2Ceil(bankCount))
  val addressBits:      Int             = log2Ceil(entriesPerBank)
  val channelBits:      Int             = math.max(1, log2Ceil(totalChannels + 1))
  val coreBits:         Int             = math.max(1, log2Ceil(cores.size))
  val stagingBank:      Int             = stagingBankId
  val stagingAddress:   Int             = entriesPerBank - 1
  val maskBits:         Int             = dataBits / 8

  // Core order in this sequence defines the flattened io.channels layout.
  def channelsForCore(coreIndex: Int): Range = {
    require(coreIndex >= 0 && coreIndex < cores.size)
    val start = cores.take(coreIndex).map(_.bankIds.size).sum
    start until (start + cores(coreIndex).bankIds.size)
  }

}

object MeshSharedMemParams {

  // Experimental 3x4 mesh: type 0/1/2 use 1/2/4 channels, with 2/2/1 cores.
  private val prototypeGlobal = GlobalConfig().copy(
    memDomain = MemDomainParam().copy(bankWidth = 128, bankEntries = 2048, bankMaskLen = 16)
  )

  val prototype: MeshSharedMemParams = MeshSharedMemParams(
    global = prototypeGlobal,
    rows = 3,
    cols = 4,
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
