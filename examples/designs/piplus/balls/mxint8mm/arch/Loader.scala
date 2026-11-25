package examples.designs.piplus.balls.mxint8mm

import framework.generic.contract.blink.BlinkParam

import framework.generic.memory.bank.SramReadIO

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/** The operands of one K block of one output tile, as the array consumes them. */
class Slot(rows: Int, cols: Int) extends Bundle {
  val full       = Bool()
  val first      = Bool()
  val last       = Bool()
  val activeRows = UInt(log2Up(rows + 1).W)
  val activeCols = UInt(log2Up(cols + 1).W)
  val cStart     = UInt(32.W)
  val aCodes     = Vec(rows, Vec(2, UInt(128.W)))
  val aScale     = Vec(rows, UInt(8.W))
  val bCodes     = Vec(cols, Vec(2, UInt(128.W)))
  val bScale     = Vec(cols, UInt(8.W))
  // The old C of the tile, read for its first block when accumulating: four columns a bank row.
  val old        = Vec(rows * cols / 4, UInt(128.W))
  // The bias of the tile's columns, read for its first block with bias.
  val bias       = Vec(cols / 4, UInt(128.W))
}

/**
 * Streams the operands of the output tiles, row-major, block after block, into two slots: port A reads the old C
 * (first block of a tile, when accumulating), the bias of its columns (first block, with bias) and then the A rows, port B the B rows, one bank row a cycle each. A
 * packed row's exponent row serves 16 blocks: it is read with the first of them and kept.
 */
@instantiable
class Loader(link: BlinkParam, rows: Int, cols: Int) extends Module {

  @public
  val io = IO(new Bundle {
    val start      = Input(Bool())
    val m          = Input(UInt(10.W))
    val n          = Input(UInt(10.W))
    val k          = Input(UInt(16.W))
    val stride     = Input(UInt(32.W))
    val aBase      = Input(UInt(32.W))
    val cPitch     = Input(UInt(32.W))
    val cBase      = Input(UInt(32.W))
    val accumulate = Input(Bool())
    val bias       = Input(Bool())
    val readA      = Flipped(new SramReadIO(link.bankWidth))
    val readB      = Flipped(new SramReadIO(link.bankWidth))
    val slots      = Output(Vec(2, new Slot(rows, cols)))
    val free       = Input(Vec(2, Bool()))
    val finished   = Output(Bool())
    // Port A is reading the old C: the pre-write bank of C.
    val readingOld = Output(Bool())
  })

  val slots = RegInit(VecInit(Seq.fill(2)(0.U.asTypeOf(new Slot(rows, cols)))))
  io.slots := slots
  for (s <- 0 until 2)(when(io.free(s))(slots(s).full := false.B))

  val m          = Reg(UInt(10.W))
  val n          = Reg(UInt(10.W))
  val blocks     = Reg(UInt(16.W))
  val k16        = Reg(UInt(16.W))
  val stride     = Reg(UInt(32.W))
  val cPitch     = Reg(UInt(32.W))
  val accumulate = Reg(Bool())
  val bias       = Reg(Bool())
  // The bias row: the n / 4 bank rows before C's window.
  val biasRow    = Reg(UInt(32.W))
  val running    = RegInit(false.B)

  // The block being loaded and its slot.
  val m0    = Reg(UInt(11.W))
  val n0    = Reg(UInt(11.W))
  val aTile = Reg(UInt(32.W))
  val bTile = Reg(UInt(32.W))
  val cTile = Reg(UInt(32.W))
  val block = Reg(UInt(16.W))
  val slot  = Reg(UInt(1.W))

  val activeRows = Mux(m - m0 > rows.U, rows.U, m - m0)
  val activeCols = Mux(n - n0 > cols.U, cols.U, n - n0)
  val groups     = activeCols >> 2
  val withOld    = accumulate && block === 0.U
  val withBias   = bias && block === 0.U
  // A block reads two code rows of each packed row, and the exponent row at every 16th block.
  val lastPart   = Mux(block(3, 0) === 0.U, 2.U, 1.U)
  val scaleRow   = k16 + (block >> 4)
  val byte       = Cat(block(3, 0), 0.U(3.W))
  val target     = slots(slot)
  val loading    = running && !target.full

  // Kept exponent rows: one per A row and B row of the tile.
  val aExponents = Reg(Vec(rows, UInt(128.W)))
  val bExponents = Reg(Vec(cols, UInt(128.W)))

  // Issue and response positions: old C (row, group), then A (row, part); B (row, part).
  val oldIssueRow   = Reg(UInt(log2Up(rows + 1).W))
  val oldIssueGroup = Reg(UInt(log2Up(cols / 4 + 1).W))
  val oldRespRow    = Reg(UInt(log2Up(rows + 1).W))
  val oldRespGroup  = Reg(UInt(log2Up(cols / 4 + 1).W))
  val biasIssue     = Reg(UInt(log2Up(cols / 4 + 1).W))
  val biasResp      = Reg(UInt(log2Up(cols / 4 + 1).W))
  val aIssueRow     = Reg(UInt(log2Up(rows + 1).W))
  val aIssuePart    = Reg(UInt(2.W))
  val aRespRow      = Reg(UInt(log2Up(rows + 1).W))
  val aRespPart     = Reg(UInt(2.W))
  val bIssueRow     = Reg(UInt(log2Up(cols + 1).W))
  val bIssuePart    = Reg(UInt(2.W))
  val bRespRow      = Reg(UInt(log2Up(cols + 1).W))
  val bRespPart     = Reg(UInt(2.W))

  val oldIssuing  = oldIssueRow < Mux(withOld, activeRows, 0.U)
  val oldWaiting  = oldRespRow < Mux(withOld, activeRows, 0.U)
  val biasIssuing = !oldIssuing && biasIssue < Mux(withBias, groups, 0.U)
  val biasWaiting = !oldWaiting && biasResp < Mux(withBias, groups, 0.U)

  io.readA.req.valid     := loading && (oldIssuing || biasIssuing || aIssueRow < activeRows)
  io.readA.req.bits.addr := Mux(
    oldIssuing,
    cTile + (n0 >> 2) + oldIssueRow * cPitch + oldIssueGroup,
    Mux(
      biasIssuing,
      biasRow + (n0 >> 2) + biasIssue,
      aTile + aIssueRow * stride + Mux(aIssuePart === 2.U, scaleRow, Cat(block, aIssuePart(0)))
    )
  )
  io.readA.resp.ready    := loading
  io.readB.req.valid     := loading && bIssueRow < activeCols
  io.readB.req.bits.addr := bTile + bIssueRow * stride + Mux(bIssuePart === 2.U, scaleRow, Cat(block, bIssuePart(0)))
  io.readB.resp.ready    := loading

  when(io.readA.req.fire) {
    when(oldIssuing) {
      oldIssueGroup                                    := Mux(oldIssueGroup === groups - 1.U, 0.U, oldIssueGroup + 1.U)
      when(oldIssueGroup === groups - 1.U)(oldIssueRow := oldIssueRow + 1.U)
    }.elsewhen(biasIssuing) {
      biasIssue := biasIssue + 1.U
    }.otherwise {
      aIssuePart                              := Mux(aIssuePart === lastPart, 0.U, aIssuePart + 1.U)
      when(aIssuePart === lastPart)(aIssueRow := aIssueRow + 1.U)
    }
  }
  when(io.readB.req.fire) {
    bIssuePart                              := Mux(bIssuePart === lastPart, 0.U, bIssuePart + 1.U)
    when(bIssuePart === lastPart)(bIssueRow := bIssueRow + 1.U)
  }

  val dataA = io.readA.resp.bits.data
  val dataB = io.readB.resp.bits.data
  when(io.readA.resp.fire) {
    when(oldWaiting) {
      target.old(oldRespRow * (cols / 4).U + oldRespGroup) := dataA
      oldRespGroup                                         := Mux(oldRespGroup === groups - 1.U, 0.U, oldRespGroup + 1.U)
      when(oldRespGroup === groups - 1.U)(oldRespRow       := oldRespRow + 1.U)
    }.elsewhen(biasWaiting) {
      target.bias(biasResp) := dataA
      biasResp              := biasResp + 1.U
    }.otherwise {
      when(aRespPart === 2.U) {
        aExponents(aRespRow)    := dataA
        target.aScale(aRespRow) := (dataA >> byte)(7, 0)
      }.otherwise(target.aCodes(aRespRow)(aRespPart(0))                   := dataA)
      when(aRespPart === 1.U && lastPart === 1.U)(target.aScale(aRespRow) := (aExponents(aRespRow) >> byte)(7, 0))
      aRespPart                                                           := Mux(aRespPart === lastPart, 0.U, aRespPart + 1.U)
      when(aRespPart === lastPart)(aRespRow                               := aRespRow + 1.U)
    }
  }
  when(io.readB.resp.fire) {
    when(bRespPart === 2.U) {
      bExponents(bRespRow)    := dataB
      target.bScale(bRespRow) := (dataB >> byte)(7, 0)
    }.otherwise(target.bCodes(bRespRow)(bRespPart(0))                   := dataB)
    when(bRespPart === 1.U && lastPart === 1.U)(target.bScale(bRespRow) := (bExponents(bRespRow) >> byte)(7, 0))
    bRespPart                                                           := Mux(bRespPart === lastPart, 0.U, bRespPart + 1.U)
    when(bRespPart === lastPart)(bRespRow                               := bRespRow + 1.U)
  }

  // The block is in its slot: hand it over and go on to the next block, tile by tile, row-major.
  val loaded = loading && !oldWaiting && !biasWaiting && aRespRow === activeRows && bRespRow === activeCols
  when(loaded) {
    target.full       := true.B
    target.first      := block === 0.U
    target.last       := block === blocks - 1.U
    target.activeRows := activeRows
    target.activeCols := activeCols
    target.cStart     := cTile + (n0 >> 2)
    slot              := ~slot
    oldIssueRow       := 0.U
    oldIssueGroup     := 0.U
    oldRespRow        := 0.U
    oldRespGroup      := 0.U
    biasIssue         := 0.U
    biasResp          := 0.U
    aIssueRow         := 0.U
    aIssuePart        := 0.U
    aRespRow          := 0.U
    aRespPart         := 0.U
    bIssueRow         := 0.U
    bIssuePart        := 0.U
    bRespRow          := 0.U
    bRespPart         := 0.U
    block             := block + 1.U
    when(block === blocks - 1.U) {
      block := 0.U
      when(n0 +& cols.U >= n) {
        n0    := 0.U
        bTile := 0.U
        when(m0 +& rows.U >= m)(running := false.B).otherwise {
          m0    := m0 + rows.U
          aTile := aTile + stride * rows.U
          cTile := cTile + cPitch * rows.U
        }
      }.otherwise {
        n0    := n0 + cols.U
        bTile := bTile + stride * cols.U
      }
    }
  }
  io.finished := !running
  io.readingOld := oldIssuing || biasIssuing

  when(io.start) {
    m             := io.m
    n             := io.n
    blocks        := io.k >> 5
    k16           := io.k >> 4
    stride        := io.stride
    cPitch        := io.cPitch
    accumulate    := io.accumulate
    bias          := io.bias
    biasRow       := io.cBase - (io.n >> 2)
    running       := true.B
    m0            := 0.U
    n0            := 0.U
    aTile         := io.aBase
    bTile         := 0.U
    cTile         := io.cBase
    block         := 0.U
    slot          := 0.U
    oldIssueRow   := 0.U
    oldIssueGroup := 0.U
    oldRespRow    := 0.U
    oldRespGroup  := 0.U
    biasIssue     := 0.U
    biasResp      := 0.U
    aIssueRow     := 0.U
    aIssuePart    := 0.U
    aRespRow      := 0.U
    aRespPart     := 0.U
    bIssueRow     := 0.U
    bIssuePart    := 0.U
    bRespRow      := 0.U
    bRespPart     := 0.U
  }
}
