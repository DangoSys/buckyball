package memcore.memory.mesh_shm

import chisel3.util.log2Ceil
import memcore.memory.bank.BankSetParams

case class MeshCoreAttachment(
  inputBankIds:  Seq[Int],
  outputBankIds: Seq[Int],
  localBankNode: Option[Int]) {
  require(inputBankIds.size == outputBankIds.size, "mesh core input/output channel counts must match")
}

case class MeshSharedMemParams(
  rows:             Int,
  cols:             Int,
  entriesPerBank:   Int,
  dataBits:         Int,
  tagBits:          Int,
  cores:            Seq[MeshCoreAttachment],
  maxInFlight:      Int = 4,
  linkBufferDepth:  Int = 2,
  localBankBits:    Int = 10,
  visibleBanks:     Int = 0,
  externalChannels: Int = 0) {
  require(rows > 0 && cols > 0)
  require(entriesPerBank <= 65536)
  require(entriesPerBank >= 2 && (entriesPerBank & (entriesPerBank - 1)) == 0)
  require(dataBits > 0 && dataBits % 8 == 0)
  require(tagBits > 0 && localBankBits > 0 && cores.nonEmpty)
  require(cores.forall(c =>
    (c.inputBankIds ++ c.outputBankIds ++ c.localBankNode).forall(id => id >= 0 && id < rows * cols)
  ))

  require(externalChannels >= 0)
  require(linkBufferDepth > 0)
  require(maxInFlight > 0 && maxInFlight <= (1 << tagBits))
  val bankParams = BankSetParams(dataBits, 1, entriesPerBank, tagBits)
  val bankCount:              Int             = rows * cols
  val visibleBankCount:       Int             = if (visibleBanks == 0) bankCount else visibleBanks
  require(visibleBankCount > 0 && visibleBankCount <= bankCount)
  require(cores.exists(_.inputBankIds.nonEmpty), "mesh requires a data endpoint")
  require(cores.forall(core => core.inputBankIds.nonEmpty == core.localBankNode.nonEmpty))
  val totalChannels:          Int             = cores.map(_.inputBankIds.size).sum
  val inputChannelLocations:  Seq[(Int, Int)] =
    cores.flatMap(c => c.inputBankIds.map(id => (id / cols, id % cols)))
  val outputChannelLocations: Seq[(Int, Int)] =
    cores.flatMap(c => c.outputBankIds.map(id => (id / cols, id % cols)))
  val rowBits:                Int             = math.max(1, log2Ceil(rows))
  val colBits:                Int             = math.max(1, log2Ceil(cols))
  val bankBits:               Int             = math.max(1, log2Ceil(bankCount))
  val addressBits:            Int             = 16
  val networkClientChannels:  Int             = 2 * totalChannels + externalChannels
  val transferChannel:        Int             = networkClientChannels
  val channelBits:            Int             = math.max(1, log2Ceil(networkClientChannels + 1))
  val coreBits:               Int             = math.max(1, log2Ceil(cores.size))
  val linkCreditBits:         Int             = math.max(1, log2Ceil(linkBufferDepth + 1))

  val localBankLocations: Seq[Option[(Int, Int)]] =
    cores.map(_.localBankNode.map(id => (id / cols, id % cols)))

  val eventBits:          Int                     = rowBits + colBits + channelBits + addressBits + 2
  val packetUserBits:     Int                     = eventBits + localBankBits + 9
  val maskBits:           Int                     = dataBits / 8

  // Core order in this sequence defines the flattened io.channels layout.
  def channelsForCore(coreIndex: Int): Range = {
    require(coreIndex >= 0 && coreIndex < cores.size)
    val start = cores.take(coreIndex).map(_.inputBankIds.size).sum
    start until (start + cores(coreIndex).inputBankIds.size)
  }

}

object MeshSharedMemParams {

  def dimensions(bankCount: Int): (Int, Int) = {
    require(bankCount > 1)
    val rows = (math.sqrt(bankCount.toDouble).toInt to 1 by -1).find(bankCount % _ == 0).get
    (rows, bankCount / rows)
  }

  /**
   * Assigns each channel lane to a directed chain of adjacent Mesh nodes.
   *
   * Core order defines the stages within each lane; a Core may own fewer lanes
   * than its neighbors. A two-dimensional Mesh with an even dimension uses a
   * Hamiltonian cycle, so the last stage and reused lanes remain adjacent.
   */
  def directionalAttachments(
    rows:            Int,
    cols:            Int,
    channelsPerCore: Seq[Int]
  ): Seq[MeshCoreAttachment] = {
    require(rows > 0 && cols > 0)
    require(channelsPerCore.nonEmpty && channelsPerCore.forall(_ > 0))
    val bankCount    = rows * cols
    val channelCount = channelsPerCore.sum
    val canWrap      = bankCount == 2 || rows > 1 && cols > 1 && (rows % 2 == 0 || cols % 2 == 0)
    require(bankCount > 1)
    require(
      canWrap || channelCount < bankCount,
      "a Mesh without a Hamiltonian cycle cannot wrap directional channels"
    )

    def cycle(r: Int, c: Int): Seq[(Int, Int)] = {
      require(c % 2 == 0 && r > 1)
      val firstColumn = (0 until r).map(row => (row, 0))
      val inner       = (1 until c).flatMap { col =>
        val rowOrder = if (col % 2 == 1) (r - 1 to 1 by -1) else (1 until r)
        rowOrder.map(row => (row, col))
      }
      val firstRow    = (c - 1 to 1 by -1).map(col => (0, col))
      firstColumn ++ inner ++ firstRow
    }

    val coordinates  =
      if (rows % 2 == 0 && cols > 1) cycle(cols, rows).map { case (row, col) => (col, row) }
      else if (cols % 2 == 0 && rows > 1) cycle(rows, cols)
      else (0 until rows).flatMap { row =>
        val columns = if (row % 2 == 0) (0 until cols) else (cols - 1 to 0 by -1)
        columns.map(col => (row, col))
      }
    val bankOrder    = coordinates.map { case (row, col) => row * cols + col }
    val channelOrder = (0 until channelsPerCore.max).flatMap { channel =>
      channelsPerCore.indices.collect { case core if channel < channelsPerCore(core) => (core, channel) }
    }
    val locations    = channelOrder.zipWithIndex.map { case (owner, index) =>
      owner -> (
        bankOrder(index       % bankOrder.size),
        bankOrder((index + 1) % bankOrder.size)
      )
    }.toMap

    channelsPerCore.indices.map { core =>
      val inputs  = (0 until channelsPerCore(core)).map(channel => locations(core -> channel)._1)
      val outputs = (0 until channelsPerCore(core)).map(channel => locations(core -> channel)._2)
      MeshCoreAttachment(inputs, outputs, Some(inputs.head))
    }
  }

  // Experimental 3x4 mesh: type 0/1/2 use 1/2/4 channels, with 2/2/1 cores.
  val prototype: MeshSharedMemParams = MeshSharedMemParams(
    rows = 3,
    cols = 4,
    entriesPerBank = 1024,
    dataBits = 128,
    tagBits = 8,
    cores = directionalAttachments(3, 4, Seq(1, 1, 2, 2, 4))
  )

}
