package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.backend.banks.{SramReadReq, SramReadResp}
import framework.top.GlobalConfig

@instantiable
class SystolicArrayLoad(val b: GlobalConfig) extends Module {
  private val tile          = SystolicArrayConst.Tile
  private val opRowBits     = SystolicArrayConst.OpRowBits
  private val abRows        = b.memDomain.bankEntries
  private val bankWidth     = log2Up(b.memDomain.bankNum)
  private val groupWidth    = log2Up(b.memDomain.bankNum)
  private val addrWidth     = log2Up(b.memDomain.bankEntries)
  private val rowIndexWidth = log2Ceil(tile)
  private val wsReuseTiles  = SystolicArrayConst.wsReuseTiles(b)
  private val slotCount     = 3
  private val slotWidth     = log2Ceil(slotCount)
  private val tagQueueDepth = 16
  private val rowFifoDepth  = 2

  private val ballMapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "SMatMulBall")
    .getOrElse(
      throw new IllegalArgumentException("SMatMulBall not found in config")
    )

  private val inBW = ballMapping.inBW

  require(inBW >= 2, "SystolicArrayLoad requires at least two read ports")
  require(
    b.memDomain.bankWidth == opRowBits,
    "SystolicArrayLoad expects one 16xi8 row per 128-bit bank read"
  )
  require(
    b.memDomain.bankEntries % tile == 0,
    "SystolicArrayLoad expects bankEntries to be an integer number of 16-row A/B tiles"
  )

  @public
  val io = IO(new Bundle {
    val bankReadReq  = Vec(inBW, Decoupled(new SramReadReq(b)))
    val bankReadResp = Vec(inBW, Flipped(Decoupled(new SramReadResp(b))))

    val ctrl_ld_i = Flipped(Decoupled(new SystolicCtrlLoadReq(b)))

    val load_ex_req_kind          = Output(UInt(2.W))
    val load_ex_k_tile_kind       = Output(UInt(2.W))
    val load_ex_acc_slot          = Output(UInt(log2Ceil(wsReuseTiles).W))
    val load_ex_valid_m           = Output(UInt(5.W))
    val load_ex_valid_n           = Output(UInt(5.W))
    val load_ex_valid_k           = Output(UInt(5.W))
    val load_ex_b_valid_n         = Output(UInt(5.W))
    val load_ex_b_valid_k         = Output(UInt(5.W))
    val load_ex_weight_generation = Output(Bool())
    val load_ex_op1_o             = Decoupled(UInt(opRowBits.W))
    val load_ex_op2_o             = Decoupled(UInt(opRowBits.W))

    val op1_rd_bank_o  = Output(UInt(bankWidth.W))
    val op1_rd_group_o = Output(UInt(groupWidth.W))
    val op2_rd_bank_o  = Output(UInt(bankWidth.W))
    val op2_rd_group_o = Output(UInt(groupWidth.W))
  })

  private def fitTo(x: UInt, width: Int): UInt =
    if (x.getWidth >= width) x(width - 1, 0) else x.pad(width)

  private def rowGroup(baseGroup: UInt, rowBase: UInt, rowIdx: UInt): UInt = {
    val rowLinear = rowBase.pad(16) + rowIdx.pad(16)
    fitTo(baseGroup.pad(16) + (rowLinear / abRows.U), groupWidth)
  }

  private def rowAddr(rowBase: UInt, rowIdx: UInt): UInt = {
    val rowLinear = rowBase.pad(16) + rowIdx.pad(16)
    fitTo(rowLinear % abRows.U, addrWidth)
  }

  private def needOp2(kind: UInt): Bool =
    kind === SystolicCtrlLoadReqKind.READ_AB || kind === SystolicCtrlLoadReqKind.READ_A_B_PE ||
      kind === SystolicCtrlLoadReqKind.READ_A_B_BUF

  private def nextSlot(slot: UInt): UInt =
    Mux(slot === (slotCount - 1).U, 0.U, slot + 1.U)

  class LoadSlot extends Bundle {
    val req_kind          = UInt(2.W)
    val k_tile_kind       = UInt(2.W)
    val acc_slot          = UInt(log2Ceil(wsReuseTiles).W)
    val valid_m           = UInt(5.W)
    val valid_n           = UInt(5.W)
    val valid_k           = UInt(5.W)
    val b_valid_n         = UInt(5.W)
    val b_valid_k         = UInt(5.W)
    val weight_generation = Bool()
    val op1_bank          = UInt(bankWidth.W)
    val op1_group         = UInt(groupWidth.W)
    val op1_row_base      = UInt(addrWidth.W)
    val op2_bank          = UInt(bankWidth.W)
    val op2_group         = UInt(groupWidth.W)
    val op2_row_base      = UInt(addrWidth.W)
  }

  class ReadTag extends Bundle {
    val slot = UInt(slotWidth.W)
    val row  = UInt(rowIndexWidth.W)
  }

  class RowData extends Bundle {
    val slot = UInt(slotWidth.W)
    val row  = UInt(rowIndexWidth.W)
    val data = UInt(opRowBits.W)
  }

  val slotReq = RegInit(
    VecInit(Seq.fill(slotCount)(0.U.asTypeOf(new LoadSlot)))
  )

  val slotOccupied  = RegInit(VecInit(Seq.fill(slotCount)(false.B)))
  val slotAReqCount = RegInit(VecInit(Seq.fill(slotCount)(0.U(5.W))))
  val slotBReqCount = RegInit(VecInit(Seq.fill(slotCount)(0.U(5.W))))

  val allocSlot = RegInit(0.U(slotWidth.W))
  val aReqSlot  = RegInit(0.U(slotWidth.W))
  val bReqSlot  = RegInit(0.U(slotWidth.W))
  val sendSlot  = RegInit(0.U(slotWidth.W))

  io.ctrl_ld_i.ready := !slotOccupied(allocSlot)
  when(io.ctrl_ld_i.fire) {
    val input = io.ctrl_ld_i.bits
    slotReq(allocSlot).req_kind          := input.req_kind
    slotReq(allocSlot).k_tile_kind       := input.k_tile_kind
    slotReq(allocSlot).acc_slot          := input.acc_slot
    slotReq(allocSlot).valid_m           := input.valid_m
    slotReq(allocSlot).valid_n           := input.valid_n
    slotReq(allocSlot).valid_k           := input.valid_k
    slotReq(allocSlot).b_valid_n         := input.b_valid_n
    slotReq(allocSlot).b_valid_k         := input.b_valid_k
    slotReq(allocSlot).weight_generation := input.weight_generation
    slotReq(allocSlot).op1_bank          := input.op1_bank
    slotReq(allocSlot).op1_group         := input.op1_group
    slotReq(allocSlot).op1_row_base      := input.op1_row_base
    slotReq(allocSlot).op2_bank          := input.op2_bank
    slotReq(allocSlot).op2_group         := input.op2_group
    slotReq(allocSlot).op2_row_base      := input.op2_row_base
    slotOccupied(allocSlot)              := true.B
    slotAReqCount(allocSlot)             := 0.U
    slotBReqCount(allocSlot)             := 0.U
    allocSlot                            := nextSlot(allocSlot)

    assert(input.valid_m >= 1.U && input.valid_m <= tile.U)
    assert(input.valid_n >= 1.U && input.valid_n <= tile.U)
    assert(input.valid_k >= 1.U && input.valid_k <= tile.U)
    assert(!needOp2(input.req_kind) || input.b_valid_k >= 1.U)
  }

  val sendReq        = slotReq(sendSlot)
  val sendNeedOp2    = needOp2(sendReq.req_kind)
  val sendATotalRows = sendReq.valid_m
  val sendBTotalRows = Mux(sendNeedOp2, sendReq.b_valid_k, 0.U(5.W))
  val sendARow       = RegInit(0.U(5.W))
  val sendBRow       = RegInit(0.U(5.W))
  val sendADone      = RegInit(false.B)
  val sendBDone      = RegInit(false.B)
  val sendActive     = slotOccupied(sendSlot)

  io.load_ex_req_kind          := sendReq.req_kind
  io.load_ex_k_tile_kind       := sendReq.k_tile_kind
  io.load_ex_acc_slot          := sendReq.acc_slot
  io.load_ex_valid_m           := sendReq.valid_m
  io.load_ex_valid_n           := sendReq.valid_n
  io.load_ex_valid_k           := sendReq.valid_k
  io.load_ex_b_valid_n         := sendReq.b_valid_n
  io.load_ex_b_valid_k         := sendReq.b_valid_k
  io.load_ex_weight_generation := sendReq.weight_generation

  val aTagQ = Module(new Queue(new ReadTag, entries = tagQueueDepth))
  val bTagQ = Module(new Queue(new ReadTag, entries = tagQueueDepth))
  val aRowQ = Module(new Queue(new RowData, entries = rowFifoDepth))
  val bRowQ = Module(new Queue(new RowData, entries = rowFifoDepth))

  val aReq          = slotReq(aReqSlot)
  val bReq          = slotReq(bReqSlot)
  val bNeed         = needOp2(bReq.req_kind)
  val aTotalRows    = aReq.valid_m
  val bTotalRows    = Mux(bNeed, bReq.b_valid_k, 0.U(5.W))
  val aRequestsDone = slotAReqCount(aReqSlot) >= aTotalRows
  val bRequestsDone = !bNeed || slotBReqCount(bReqSlot) >= bTotalRows
  val aCanRequest   = slotOccupied(aReqSlot) && !aRequestsDone
  val bCanRequest   = slotOccupied(bReqSlot) && !bRequestsDone

  for (i <- 0 until inBW) {
    io.bankReadReq(i).valid     := false.B
    io.bankReadReq(i).bits.addr := 0.U
    io.bankReadResp(i).ready    := false.B
  }

  io.op1_rd_bank_o  := aReq.op1_bank
  io.op1_rd_group_o := rowGroup(
    aReq.op1_group,
    aReq.op1_row_base,
    slotAReqCount(aReqSlot)
  )
  io.op2_rd_bank_o  := bReq.op2_bank
  io.op2_rd_group_o := rowGroup(
    bReq.op2_group,
    bReq.op2_row_base,
    slotBReqCount(bReqSlot)
  )

  io.bankReadReq(0).valid     := aCanRequest && aTagQ.io.enq.ready
  io.bankReadReq(0).bits.addr := rowAddr(
    aReq.op1_row_base,
    slotAReqCount(aReqSlot)
  )
  aTagQ.io.enq.valid          := io.bankReadReq(0).fire
  aTagQ.io.enq.bits.slot      := aReqSlot
  aTagQ.io.enq.bits.row       := slotAReqCount(aReqSlot)(rowIndexWidth - 1, 0)

  io.bankReadReq(1).valid     := bCanRequest && bTagQ.io.enq.ready
  io.bankReadReq(1).bits.addr := rowAddr(
    bReq.op2_row_base,
    slotBReqCount(bReqSlot)
  )
  bTagQ.io.enq.valid          := io.bankReadReq(1).fire
  bTagQ.io.enq.bits.slot      := bReqSlot
  bTagQ.io.enq.bits.row       := slotBReqCount(bReqSlot)(rowIndexWidth - 1, 0)

  when(io.bankReadReq(0).fire) {
    slotAReqCount(aReqSlot) := slotAReqCount(aReqSlot) + 1.U
    when(slotAReqCount(aReqSlot) + 1.U >= aTotalRows) {
      aReqSlot := nextSlot(aReqSlot)
    }
  }.elsewhen(slotOccupied(aReqSlot) && aRequestsDone) {
    aReqSlot := nextSlot(aReqSlot)
  }

  when(io.bankReadReq(1).fire) {
    slotBReqCount(bReqSlot) := slotBReqCount(bReqSlot) + 1.U
    when(slotBReqCount(bReqSlot) + 1.U >= bTotalRows) {
      bReqSlot := nextSlot(bReqSlot)
    }
  }.elsewhen(slotOccupied(bReqSlot) && bRequestsDone) {
    bReqSlot := nextSlot(bReqSlot)
  }

  val aResponseReady = aTagQ.io.deq.valid && aRowQ.io.enq.ready
  val bResponseReady = bTagQ.io.deq.valid && bRowQ.io.enq.ready
  io.bankReadResp(0).ready := aResponseReady
  io.bankReadResp(1).ready := bResponseReady
  aTagQ.io.deq.ready       := io.bankReadResp(0).fire
  bTagQ.io.deq.ready       := io.bankReadResp(1).fire
  aRowQ.io.enq.valid       := io.bankReadResp(0).fire
  bRowQ.io.enq.valid       := io.bankReadResp(1).fire
  aRowQ.io.enq.bits.slot   := aTagQ.io.deq.bits.slot
  aRowQ.io.enq.bits.row    := aTagQ.io.deq.bits.row
  aRowQ.io.enq.bits.data   := io.bankReadResp(0).bits.data
  bRowQ.io.enq.bits.slot   := bTagQ.io.deq.bits.slot
  bRowQ.io.enq.bits.row    := bTagQ.io.deq.bits.row
  bRowQ.io.enq.bits.data   := io.bankReadResp(1).bits.data

  val aHeadMatches = aRowQ.io.deq.valid &&
    aRowQ.io.deq.bits.slot === sendSlot &&
    aRowQ.io.deq.bits.row === sendARow(rowIndexWidth - 1, 0)

  val bHeadMatches = bRowQ.io.deq.valid &&
    bRowQ.io.deq.bits.slot === sendSlot &&
    bRowQ.io.deq.bits.row === sendBRow(rowIndexWidth - 1, 0)

  io.load_ex_op1_o.valid := sendActive && !sendADone && aHeadMatches
  io.load_ex_op1_o.bits  := aRowQ.io.deq.bits.data
  io.load_ex_op2_o.valid := sendActive && sendNeedOp2 && !sendBDone && bHeadMatches
  io.load_ex_op2_o.bits  := bRowQ.io.deq.bits.data
  aRowQ.io.deq.ready     := io.load_ex_op1_o.fire
  bRowQ.io.deq.ready     := io.load_ex_op2_o.fire

  val op1Fire            = io.load_ex_op1_o.fire
  val op2Fire            = io.load_ex_op2_o.fire
  val aFinishesThisCycle = op1Fire && sendARow + 1.U >= sendATotalRows
  val bFinishesThisCycle = op2Fire && sendBRow + 1.U >= sendBTotalRows
  val aDoneNext          = sendADone || aFinishesThisCycle
  val bDoneNext          = !sendNeedOp2 || sendBDone || bFinishesThisCycle

  when(op1Fire) {
    when(aFinishesThisCycle) {
      sendADone := true.B
    }.otherwise {
      sendARow := sendARow + 1.U
    }
  }
  when(op2Fire) {
    when(bFinishesThisCycle) {
      sendBDone := true.B
    }.otherwise {
      sendBRow := sendBRow + 1.U
    }
  }

  when(sendActive && aDoneNext && bDoneNext) {
    slotOccupied(sendSlot) := false.B
    sendSlot               := nextSlot(sendSlot)
    sendARow               := 0.U
    sendBRow               := 0.U
    sendADone              := false.B
    sendBDone              := false.B
  }

  for (slot <- 0 until slotCount) {
    val slotNeedB = needOp2(slotReq(slot).req_kind)
    when(slotOccupied(slot)) {
      assert(slotAReqCount(slot) <= slotReq(slot).valid_m)
      assert(
        slotBReqCount(slot) <= Mux(slotNeedB, slotReq(slot).b_valid_k, 0.U)
      )
    }
  }
  when(io.bankReadResp(0).valid) {
    assert(
      aTagQ.io.deq.valid,
      "SystolicArrayLoad: A response arrived without an outstanding request"
    )
  }
  when(io.bankReadResp(1).valid) {
    assert(
      bTagQ.io.deq.valid,
      "SystolicArrayLoad: B response arrived without an outstanding request"
    )
  }
}
