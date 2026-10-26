package framework.ip.memory.mesh_shm

import chisel3.util.log2Ceil
import framework.ip.memory.bank.BankSetParams

case class MeshThreadAttachment(bank: Int, ports: Int)

case class MeshSharedMemParams(
  rows:             Int,
  cols:             Int,
  entriesPerBank:   Int,
  dataBits:         Int,
  tagBits:          Int,
  threads:          Seq[MeshThreadAttachment],
  maxInFlight:      Int = 4,
  localBankBits:    Int = 10,
  visibleBanks:     Int = 0,
  externalChannels: Int = 0,
  linkBufferDepth:  Int = 2) {
  require(rows > 0 && cols > 0)
  require(entriesPerBank <= 65536)
  require(entriesPerBank >= 2 && (entriesPerBank & (entriesPerBank - 1)) == 0)
  require(dataBits > 0 && dataBits % 8 == 0)
  require(tagBits > 0 && localBankBits > 0 && threads.nonEmpty)
  require(threads.forall(t => t.bank >= 0 && t.bank < rows * cols && t.ports >= 0))

  require(externalChannels >= 0)
  require(linkBufferDepth > 0)
  val linkCreditBits: Int = math.max(1, log2Ceil(linkBufferDepth + 1))
  require(maxInFlight > 0 && maxInFlight <= (1 << tagBits))
  val bankParams = BankSetParams(dataBits, 1, entriesPerBank, tagBits)
  val bankCount:        Int             = rows * cols
  val visibleBankCount: Int             = if (visibleBanks == 0) bankCount else visibleBanks
  require(visibleBankCount > 0 && visibleBankCount <= bankCount)
  require(threads.exists(_.ports > 0), "mesh requires a data endpoint")
  val totalChannels:    Int             = threads.map(_.ports).sum
  val channelLocations: Seq[(Int, Int)] =
    threads.flatMap(t => Seq.fill(t.ports)((t.bank / cols, t.bank % cols)))
  val rowBits:          Int             = math.max(1, log2Ceil(rows))
  val colBits:          Int             = math.max(1, log2Ceil(cols))
  val bankBits:         Int             = math.max(1, log2Ceil(bankCount))
  val addressBits:      Int             = 16
  val channelBits:      Int             = math.max(1, log2Ceil(totalChannels + 1 + externalChannels))
  val threadBits:       Int             = math.max(1, log2Ceil(threads.size))

  val threadLocations: Seq[(Int, Int)] = threads.map(t => (t.bank / cols, t.bank % cols))

  val eventBits:      Int = rowBits + colBits + channelBits + addressBits + 2
  val packetUserBits: Int = eventBits + localBankBits + 9
  val maskBits:       Int = dataBits / 8

  // Thread order in this sequence defines the flattened io.channels layout.
  def channelsForThread(threadIndex: Int): Range = {
    require(threadIndex >= 0 && threadIndex < threads.size)
    val start = threads.take(threadIndex).map(_.ports).sum
    start until (start + threads(threadIndex).ports)
  }

}

object MeshSharedMemParams {

  // Five fixed thread attachments on a 3x4 mesh, with 1/1/2/2/4 data ports.
  val prototype: MeshSharedMemParams = MeshSharedMemParams(
    rows = 3,
    cols = 4,
    entriesPerBank = 1024,
    dataBits = 128,
    tagBits = 8,
    threads = Seq(
      MeshThreadAttachment(0, 1),
      MeshThreadAttachment(3, 1),
      MeshThreadAttachment(4, 2),
      MeshThreadAttachment(7, 2),
      MeshThreadAttachment(5, 4)
    )
  )

}
