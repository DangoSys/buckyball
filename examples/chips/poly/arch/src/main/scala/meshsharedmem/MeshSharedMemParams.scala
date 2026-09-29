package examples.poly.meshsharedmem

import chisel3.util.log2Ceil

case class MeshCoreAttachment(bankIds: Seq[Int])

case class MeshSharedMemParams(
  rows:           Int,
  cols:           Int,
  entriesPerBank: Int,
  dataBits:       Int,
  tagBits:        Int,
  cores:          Seq[MeshCoreAttachment]) {
  require(rows > 0 && cols > 0)
  require(entriesPerBank >= 2 && (entriesPerBank & (entriesPerBank - 1)) == 0)
  require(dataBits > 0 && dataBits % 8 == 0)
  require(tagBits > 0 && cores.nonEmpty)
  require(cores.forall(c => c.bankIds.nonEmpty && c.bankIds.forall(id => id >= 0 && id < rows * cols)))

  val bankCount:        Int             = rows * cols
  val totalChannels:    Int             = cores.map(_.bankIds.size).sum
  val channelLocations: Seq[(Int, Int)] =
    cores.flatMap(c => c.bankIds.map(id => (id / cols, id % cols)))
  val rowBits:          Int             = math.max(1, log2Ceil(rows))
  val colBits:          Int             = math.max(1, log2Ceil(cols))
  val bankBits:         Int             = math.max(1, log2Ceil(bankCount))
  val addressBits:      Int             = log2Ceil(entriesPerBank)
  val channelBits:      Int             = math.max(1, log2Ceil(totalChannels + 1))
  val coreBits:         Int             = math.max(1, log2Ceil(cores.size))
  val stagingBank:      Int             = 0
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
  val prototype: MeshSharedMemParams = MeshSharedMemParams(
    rows = 3,
    cols = 4,
    entriesPerBank = 2048,
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
