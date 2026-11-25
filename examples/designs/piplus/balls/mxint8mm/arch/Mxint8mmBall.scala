package examples.designs.piplus.balls.mxint8mm

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.mxint8mm.{Params => ParametersProto}
import framework.ball.blink.{BlinkIO, HasBlink}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}

/**
 * MXINT8MM: C[M,N] (FP32) = A[M,K] . B[N,K]^T over packed MXINT8 rows, output tiles of rows x cols row-major. The
 * Loader streams each K block of a tile into one of two slots; a row of `cols` columns walks the rows of a loaded
 * block, `steps` MAC cycles a row, then the conversion of the row's block sums to FP32 terms (and, in the tile's first
 * block, the start of its accumulators: +0.0 or the old C, plus the bias) and their accumulation. The C rows of a tile
 * follow the rows of its last block to the bank, one bank row (four columns) a write.
 */
@instantiable
class Mxint8mmBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  val params = ball.parameters.unpack(classOf[ParametersProto])
  require(params.hasRows && params.hasCols && params.hasLanes, "Mxint8mmBall requires rows, cols and lanes")
  val rows   = params.getRows
  val cols   = params.getCols
  val lanes  = params.getLanes
  require(rows > 0 && cols > 0 && cols   % 4 == 0, "Mxint8mmBall needs a positive array with cols a multiple of 4")
  require(lanes > 0 && lanes <= 16 && 32 % lanes == 0, "Mxint8mmBall lanes must divide 32, two MAC cycles a row or more")
  require(link.bankWidth == 128 && link.bankMaskLen == 16, "Mxint8mmBall requires 128-bit bank rows and byte masks")
  require(ball.inBW == 2 && ball.outBW == 1, "Mxint8mmBall requires two reads and one write")
  require(link.dataBits == 64, "Mxint8mmBall decodes a 64-bit rs2")
  val steps = 32 / lanes

  @public
  val blink = IO(new BlinkIO(link, route))

  val busy       = RegInit(false.B)
  val done       = RegInit(false.B)
  val robId      = Reg(UInt(link.robIdBits.W))
  val aBank      = Reg(UInt(link.bankIdBits.W))
  val bBank      = Reg(UInt(link.bankIdBits.W))
  val cBank      = Reg(UInt(link.bankIdBits.W))
  val oldBank    = Reg(UInt(link.bankIdBits.W))
  val aGroup     = Reg(UInt(link.groupIdBits.W))
  val bGroup     = Reg(UInt(link.groupIdBits.W))
  val cGroup     = Reg(UInt(link.groupIdBits.W))
  val cPitch     = Reg(UInt(14.W))
  val accumulate = Reg(Bool())
  val bias       = Reg(Bool())

  val active = !reset.asBool
  blink.cmdReq.ready            := !busy && !done && active
  blink.cmdResp.valid           := done && active
  blink.cmdResp.bits.rob_id     := robId
  blink.status.idle             := !busy && !done
  blink.status.running          := busy
  when(blink.cmdResp.fire)(done := false.B)

  val loader = Instantiate(new Loader(link, rows, cols))
  val cmd    = blink.cmdReq.bits
  val count  = cmd.iter
  loader.io.start      := blink.cmdReq.fire
  loader.io.m          := cmd.rs2(9, 0)
  loader.io.n          := cmd.rs2(19, 10)
  loader.io.k          := count
  loader.io.stride     := (count >> 4) +& (count >> 9) +& (count(8, 0) =/= 0.U)
  loader.io.aBase      := cmd.rs2(33, 20)
  loader.io.cPitch     := cmd.rs2(47, 34)
  loader.io.cBase      := cmd.rs2(61, 48)
  loader.io.accumulate := cmd.rs2(62)
  loader.io.bias       := cmd.rs2(63)

  // Port A reads the old C from the pre-write bank of C, then A; port B reads B.
  val readA = blink.bankRead(0)
  readA.rob_id   := robId
  readA.bank_id  := Mux(loader.io.readingOld, oldBank, aBank)
  readA.group_id := Mux(loader.io.readingOld, cGroup, aGroup)
  readA.io <> loader.io.readA
  val readB = blink.bankRead(1)
  readB.rob_id   := robId
  readB.bank_id  := bBank
  readB.group_id := bGroup
  readB.io <> loader.io.readB

  val columns      = Seq.fill(cols)(Instantiate(new Column(lanes)))
  val accumulators = Reg(Vec(rows, Vec(cols, UInt(32.W))))

  // The store of a tile: its C rows, as its last block accumulates them.
  val storing     = RegInit(false.B)
  val storeStart  = Reg(UInt(32.W))
  val storeRows   = Reg(UInt(log2Up(rows + 1).W))
  val storeGroups = Reg(UInt(log2Up(cols / 4 + 1).W))
  val storeRow    = Reg(UInt(log2Up(rows + 1).W))
  val storeGroup  = Reg(UInt(log2Up(cols / 4 + 1).W))
  val added       = Reg(UInt(log2Up(rows + 1).W))
  val writes      = RegInit(0.U(16.W))

  // MAC: the rows of the block in `slot`. A tile's last block starts once the store of the tile before has ended, its
  // first block stays behind the rows that store still has to write.
  val slot  = RegInit(0.U(1.W))
  val row   = RegInit(0.U(log2Up(rows).W))
  val step  = RegInit(0.U(log2Up(steps).W))
  val cur   = loader.io.slots(slot)
  val begun = row =/= 0.U || step =/= 0.U
  val mac   = busy && cur.full && Mux(cur.last, begun || !storing, !cur.first || !storing || row < storeRow)
  val ends  = step === (steps - 1).U
  when(mac) {
    step := Mux(ends, 0.U, step + 1.U)
    when(ends) {
      row := row + 1.U
      when(row === cur.activeRows - 1.U) {
        row  := 0.U
        slot := ~slot
      }
    }
  }
  when(mac && cur.last && !begun) {
    storing     := true.B
    storeStart  := cur.cStart
    storeRows   := cur.activeRows
    storeGroups := cur.activeCols >> 2
    storeRow    := 0.U
    storeGroup  := 0.U
    added       := 0.U
  }

  // Convert: the row MAC has summed, with the operands of its slot; the slot goes back with its last row.
  val converting  = RegNext(mac && ends, false.B)
  val convertSlot = RegNext(slot)
  val convertRow  = RegNext(row)
  val converted   = loader.io.slots(convertSlot)
  val starts      = converting && converted.first
  loader.io.free(0) := converting && convertRow === converted.activeRows - 1.U && convertSlot === 0.U
  loader.io.free(1) := converting && convertRow === converted.activeRows - 1.U && convertSlot === 1.U

  // Add: the terms of the row into its accumulators.
  val adding  = RegNext(converting, false.B)
  val addRow  = RegNext(convertRow)
  val addCols = RegNext(converted.activeCols)
  val addLast = RegNext(converted.last)
  when(adding && addLast)(added := added + 1.U)

  // The `lanes` codes of the current step: bytes [step * lanes, (step + 1) * lanes) of the 32-code block.
  val aCodes = Cat(cur.aCodes(row)(1), cur.aCodes(row)(0)).asTypeOf(Vec(steps, UInt((lanes * 8).W)))(step)
  val old    = VecInit(converted.old.grouped(cols / 4).map(groups => Cat(groups.reverse)).toSeq)(convertRow)
  for (j <- 0 until cols) {
    val column = columns(j)
    // Each element starts from +0.0 or from the old C element; with bias, bias[col] next.
    val start  = Mux(accumulate, old(32 * j + 31, 32 * j), 0.U)
    val offset = converted.bias(j / 4)(32 * (j % 4) + 31, 32 * (j % 4))
    column.io.mac                            := mac
    column.io.first                          := step === 0.U
    column.io.a                              := aCodes
    column.io.b                              := Cat(cur.bCodes(j)(1), cur.bCodes(j)(0)).asTypeOf(Vec(steps, UInt((lanes * 8).W)))(step)
    column.io.convert                        := converting
    column.io.active                         := j.U < converted.activeCols
    column.io.scale                          := converted.aScale(convertRow).zext +& converted.bScale(j).zext - 254.S
    column.io.x                              := Mux(adding, accumulators(addRow)(j), start)
    column.io.y                              := Mux(adding, column.io.term, offset)
    when(starts)(accumulators(convertRow)(j) := Mux(bias, column.io.out, start))
    when(adding)(accumulators(addRow)(j)     := column.io.out)
    when(starts && column.io.active) {
      when(accumulate)(assert(start(30, 23) =/= 255.U, "MXINT8MM old C must be finite"))
      when(bias)(assert(offset(30, 23) =/= 255.U, "MXINT8MM bias must be finite"))
    }
    when((adding && j.U < addCols) || (starts && bias && column.io.active))(
      assert(column.io.out(30, 23) =/= 255.U, "MXINT8MM accumulation overflows FP32")
    )
  }

  val write     = blink.bankWrite(0)
  val lastGroup = storeGroup === storeGroups - 1.U
  write.rob_id                                          := robId
  write.bank_id                                         := cBank
  write.group_id                                        := cGroup
  write.io.req.valid                                    := storing && storeRow < added && active
  write.io.req.bits.addr                                := storeStart + storeRow * cPitch + storeGroup
  write.io.req.bits.data                                := accumulators(storeRow).asUInt.asTypeOf(Vec(cols / 4, UInt(128.W)))(storeGroup)
  write.io.req.bits.mask                                := VecInit(Seq.fill(16)(true.B))
  write.io.resp.ready                                   := active
  when(write.io.req.fire) {
    storeGroup := Mux(lastGroup, 0.U, storeGroup + 1.U)
    when(lastGroup) {
      storeRow                                   := storeRow + 1.U
      when(storeRow === storeRows - 1.U)(storing := false.B)
    }
  }
  when(write.io.req.fire =/= write.io.resp.fire)(writes := Mux(write.io.req.fire, writes + 1.U, writes - 1.U))
  when(write.io.resp.fire)(assert(write.io.resp.bits.ok, "MXINT8MM bank write failed"))

  when(busy && loader.io.finished && !loader.io.slots(0).full && !loader.io.slots(1).full && !converting && !adding &&
    !storing && writes === 0.U) {
    busy := false.B
    done := true.B
  }

  when(blink.cmdReq.fire) {
    val mCount = cmd.rs2(9, 0)
    val nCount = cmd.rs2(19, 10)
    val packed = loader.io.stride
    assert(mCount =/= 0.U && nCount =/= 0.U && nCount(1, 0) === 0.U, "MXINT8MM M > 0 and N a positive multiple of 4")
    assert(count =/= 0.U && count(4, 0) === 0.U, "MXINT8MM K must be a positive multiple of 32")
    assert(
      cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank,
      "MXINT8MM banks must differ"
    )
    assert(cmd.op1_col === 1.U && cmd.op2_col === 1.U && cmd.wr_col === 1.U, "MXINT8MM banks must have one group")
    assert(cmd.rs2(33, 20) +& mCount * packed <= link.bankEntries.U, "MXINT8MM A window exceeds capacity")
    assert(nCount * packed <= link.bankEntries.U, "MXINT8MM B window exceeds capacity")
    assert(cmd.rs2(47, 34) >= (nCount >> 2), "MXINT8MM C rows overlap")
    assert(
      cmd.rs2(61, 48) +& (mCount - 1.U) * cmd.rs2(47, 34) +& (nCount >> 2) <= link.bankEntries.U && (!cmd.rs2(
        63
      ) || cmd.rs2(61, 48) >= (nCount >> 2)),
      "MXINT8MM C window exceeds capacity or leaves no room for the bias before it"
    )
    busy       := true.B
    // The Loader fills slot 0 first on every command; a command of an odd block count left slot at 1.
    slot       := 0.U
    robId      := cmd.rob_id
    aBank      := cmd.op1_bank
    bBank      := cmd.op2_bank
    cBank      := cmd.wr_bank
    oldBank    := cmd.wr_read_bank
    aGroup     := cmd.op1_group
    bGroup     := cmd.op2_group
    cGroup     := cmd.wr_group
    cPitch     := cmd.rs2(47, 34)
    accumulate := cmd.rs2(62)
    bias       := cmd.rs2(63)
  }
}

object Mxint8mmBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new Mxint8mmBall(link, route)).blink, None)

}
