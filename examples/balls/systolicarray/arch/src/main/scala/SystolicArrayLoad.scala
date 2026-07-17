package examples.balls.systolicarray

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.backend.banks.{SramReadReq, SramReadResp}
import framework.top.GlobalConfig

@instantiable
class SystolicArrayLoad(val b: GlobalConfig) extends Module {
  private val tile      = SystolicArrayConst.Tile
  private val opRowBits = SystolicArrayConst.OpRowBits
  private val abRows    = b.memDomain.bankEntries
  private val bankWidth = log2Up(b.memDomain.bankNum)
  private val groupWidth = log2Up(b.memDomain.bankNum)
  private val addrWidth = log2Up(b.memDomain.bankEntries)
  private val rowIndexWidth = log2Ceil(tile)

  private val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "SystolicArrayBall")
    .getOrElse(throw new IllegalArgumentException("SystolicArrayBall not found in config"))
  private val inBW = ballMapping.inBW

  require(inBW >= 2, "SystolicArrayLoad requires at least two read ports")
  require(b.memDomain.bankWidth == opRowBits, "SystolicArrayLoad expects one 16xi8 row per 128-bit bank read")
  require(
    b.memDomain.bankEntries % tile == 0,
    "SystolicArrayLoad expects bankEntries to be an integer number of 16-row A/B tiles"
  )

  @public
  val io = IO(new Bundle {
    val bankReadReq  = Vec(inBW, Decoupled(new SramReadReq(b)))
    val bankReadResp = Vec(inBW, Flipped(Decoupled(new SramReadResp(b))))

    val ctrl_ld_i = Flipped(Decoupled(new SystolicCtrlLoadReq(b)))

    val load_ex_req_kind   = Output(UInt(2.W))
    val load_ex_k_tile_kind = Output(UInt(2.W))
    val load_ex_acc_slot    = Output(UInt(log2Ceil(SystolicArrayConst.WsReuseTiles).W))
    val load_ex_valid_m    = Output(UInt(5.W))
    val load_ex_valid_n    = Output(UInt(5.W))
    val load_ex_valid_k    = Output(UInt(5.W))
    val load_ex_row_count  = Output(UInt(5.W))
    val load_ex_first_row  = Output(Bool())
    val load_ex_last_row   = Output(Bool())
    val load_ex_op1_o      = Decoupled(UInt(opRowBits.W))
    val load_ex_op2_o      = Decoupled(UInt(opRowBits.W))

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

  private def needOp1(kind: UInt): Bool =
    kind === SystolicCtrlLoadReqKind.READ_AB || kind === SystolicCtrlLoadReqKind.READ_A_ONLY ||
      kind === SystolicCtrlLoadReqKind.READ_A_B_PE

  private def needOp2(kind: UInt): Bool =
    kind === SystolicCtrlLoadReqKind.READ_AB || kind === SystolicCtrlLoadReqKind.READ_A_B_PE

  private val slotCount = 2

  val slotReq       = RegInit(VecInit(Seq.fill(slotCount)(0.U.asTypeOf(new SystolicCtrlLoadReq(b)))))
  val slotOccupied  = RegInit(VecInit(Seq.fill(slotCount)(false.B)))
  val slotBankDone  = RegInit(VecInit(Seq.fill(slotCount)(false.B)))
  val slotFirstSent = RegInit(VecInit(Seq.fill(slotCount)(false.B)))
  val slotFillStarted = RegInit(VecInit(Seq.fill(slotCount)(false.B)))
  val bufferInUse   = RegInit(VecInit(Seq.fill(slotCount)(false.B)))
  val allocSlot     = RegInit(0.U(1.W))
  val fillSlot      = RegInit(0.U(1.W))
  val metaSlot      = RegInit(0.U(1.W))

  io.ctrl_ld_i.ready := !slotOccupied(allocSlot)
  when(io.ctrl_ld_i.fire) {
    slotReq(allocSlot)       := io.ctrl_ld_i.bits
    slotOccupied(allocSlot)  := true.B
    slotBankDone(allocSlot)  := false.B
    slotFirstSent(allocSlot) := false.B
    slotFillStarted(allocSlot) := false.B
    allocSlot := allocSlot ^ 1.U
    assert(io.ctrl_ld_i.bits.valid_m >= 1.U && io.ctrl_ld_i.bits.valid_m <= tile.U)
    assert(io.ctrl_ld_i.bits.valid_n >= 1.U && io.ctrl_ld_i.bits.valid_n <= tile.U)
    assert(io.ctrl_ld_i.bits.valid_k >= 1.U && io.ctrl_ld_i.bits.valid_k <= tile.U)
    assert(io.ctrl_ld_i.bits.row_count >= 1.U && io.ctrl_ld_i.bits.row_count <= tile.U)
  }

  val op1Rows = Reg(Vec(slotCount, Vec(tile, UInt(opRowBits.W))))
  val op2Rows = Reg(Vec(slotCount, Vec(tile, UInt(opRowBits.W))))
  val op1RowsReady = RegInit(VecInit(Seq.fill(slotCount)(0.U(5.W))))
  val op2RowsReady = RegInit(VecInit(Seq.fill(slotCount)(0.U(5.W))))

  val sendRow         = RegInit(0.U(5.W))
  val sendBuffer      = RegInit(0.U(1.W))
  val sendReqKind     = RegInit(SystolicCtrlLoadReqKind.READ_AB)
  val sendKTileKind   = RegInit(SystolicKTileKind.DIRECT)
  val sendAccSlot     = RegInit(0.U(log2Ceil(SystolicArrayConst.WsReuseTiles).W))
  val sendValidM      = RegInit(tile.U(5.W))
  val sendValidN      = RegInit(tile.U(5.W))
  val sendValidK      = RegInit(tile.U(5.W))
  val sendRowCount    = RegInit(tile.U(5.W))
  val firstSendRow    = sendRow === 0.U
  val sendMetaPresent = slotOccupied(metaSlot) && !slotFirstSent(metaSlot) &&
    slotFillStarted(metaSlot) && bufferInUse(metaSlot)
  val currentBuffer   = Mux(firstSendRow, metaSlot, sendBuffer)
  val currentReqKind  = Mux(firstSendRow, slotReq(metaSlot).req_kind, sendReqKind)
  val currentKTileKind = Mux(firstSendRow, slotReq(metaSlot).k_tile_kind, sendKTileKind)
  val currentAccSlot  = Mux(firstSendRow, slotReq(metaSlot).acc_slot, sendAccSlot)
  val currentValidM   = Mux(firstSendRow, slotReq(metaSlot).valid_m, sendValidM)
  val currentValidN   = Mux(firstSendRow, slotReq(metaSlot).valid_n, sendValidN)
  val currentValidK   = Mux(firstSendRow, slotReq(metaSlot).valid_k, sendValidK)
  val currentRowCount = Mux(firstSendRow, slotReq(metaSlot).row_count, sendRowCount)
  val currentNeedOp1  = needOp1(currentReqKind)
  val currentNeedOp2  = needOp2(currentReqKind)
  val sendActive      = !firstSendRow || sendMetaPresent

  io.load_ex_req_kind    := currentReqKind
  io.load_ex_k_tile_kind := currentKTileKind
  io.load_ex_acc_slot    := currentAccSlot
  io.load_ex_valid_m     := currentValidM
  io.load_ex_valid_n     := currentValidN
  io.load_ex_valid_k     := currentValidK
  io.load_ex_row_count   := currentRowCount
  io.load_ex_first_row   := sendActive && firstSendRow
  io.load_ex_last_row    := sendActive && sendRow + 1.U >= currentRowCount
  io.load_ex_op1_o.valid := sendActive && currentNeedOp1 && op1RowsReady(currentBuffer) > sendRow
  io.load_ex_op1_o.bits  := op1Rows(currentBuffer)(sendRow(rowIndexWidth - 1, 0))
  io.load_ex_op2_o.valid := sendActive && currentNeedOp2 && op2RowsReady(currentBuffer) > sendRow
  io.load_ex_op2_o.bits  := op2Rows(currentBuffer)(sendRow(rowIndexWidth - 1, 0))

  val op1Fire = io.load_ex_op1_o.fire
  val op2Fire = io.load_ex_op2_o.fire
  val rowDone = MuxLookup(currentReqKind, false.B)(Seq(
    SystolicCtrlLoadReqKind.READ_AB     -> (op1Fire && op2Fire),
    SystolicCtrlLoadReqKind.READ_A_ONLY -> op1Fire,
    SystolicCtrlLoadReqKind.READ_A_B_PE -> (op1Fire && op2Fire)
  ))
  when(rowDone) {
    when(firstSendRow) {
      sendReqKind   := slotReq(metaSlot).req_kind
      sendKTileKind := slotReq(metaSlot).k_tile_kind
      sendAccSlot   := slotReq(metaSlot).acc_slot
      sendValidM    := slotReq(metaSlot).valid_m
      sendValidN    := slotReq(metaSlot).valid_n
      sendValidK    := slotReq(metaSlot).valid_k
      sendRowCount  := slotReq(metaSlot).row_count
      sendBuffer    := metaSlot
      slotFirstSent(metaSlot) := true.B
      when(slotBankDone(metaSlot)) {
        slotOccupied(metaSlot) := false.B
      }
      metaSlot := metaSlot ^ 1.U
    }

    when(sendRow + 1.U >= currentRowCount) {
      sendRow := 0.U
      bufferInUse(currentBuffer) := false.B
    }.otherwise {
      sendRow := sendRow + 1.U
    }
  }

  val op1ReqRow    = RegInit(0.U(5.W))
  val op2ReqRow    = RegInit(0.U(5.W))
  val op1RespCount = RegInit(0.U(5.W))
  val op2RespCount = RegInit(0.U(5.W))
  val fillReq      = slotReq(fillSlot)
  val fillCanStart = slotOccupied(fillSlot) && !slotBankDone(fillSlot) &&
    !slotFillStarted(fillSlot) && !bufferInUse(fillSlot)
  val fillActive   = slotOccupied(fillSlot) && slotFillStarted(fillSlot) && !slotBankDone(fillSlot)
  val fillNeedOp1  = needOp1(fillReq.req_kind)
  val fillNeedOp2  = needOp2(fillReq.req_kind)
  val fillRowCount = fillReq.row_count

  when(fillCanStart) {
    slotFillStarted(fillSlot) := true.B
    bufferInUse(fillSlot) := true.B
    op1RowsReady(fillSlot) := 0.U
    op2RowsReady(fillSlot) := 0.U
    op1ReqRow := 0.U
    op2ReqRow := 0.U
    op1RespCount := 0.U
    op2RespCount := 0.U
  }

  for (i <- 0 until inBW) {
    io.bankReadReq(i).valid     := false.B
    io.bankReadReq(i).bits.addr := 0.U
    io.bankReadResp(i).ready    := false.B
  }

  io.op1_rd_bank_o  := fillReq.op1_bank
  io.op1_rd_group_o := rowGroup(fillReq.op1_group, fillReq.op1_row_base, op1ReqRow)
  io.op2_rd_bank_o  := fillReq.op2_bank
  io.op2_rd_group_o := rowGroup(fillReq.op2_group, fillReq.op2_row_base, op2ReqRow)

  when(fillActive && fillNeedOp1 && op1ReqRow < fillRowCount) {
    io.bankReadReq(0).valid     := true.B
    io.bankReadReq(0).bits.addr := rowAddr(fillReq.op1_row_base, op1ReqRow)
    when(io.bankReadReq(0).fire) {
      op1ReqRow := op1ReqRow + 1.U
    }
  }

  when(fillActive && fillNeedOp2 && op2ReqRow < fillRowCount) {
    io.bankReadReq(1).valid     := true.B
    io.bankReadReq(1).bits.addr := rowAddr(fillReq.op2_row_base, op2ReqRow)
    when(io.bankReadReq(1).fire) {
      op2ReqRow := op2ReqRow + 1.U
    }
  }

  io.bankReadResp(0).ready := fillActive && fillNeedOp1 && op1RespCount < fillRowCount
  io.bankReadResp(1).ready := fillActive && fillNeedOp2 && op2RespCount < fillRowCount

  val op1Enq = io.bankReadResp(0).fire
  val op2Enq = io.bankReadResp(1).fire

  when(op1Enq) {
    op1Rows(fillSlot)(op1RespCount(rowIndexWidth - 1, 0)) := io.bankReadResp(0).bits.data
    op1RowsReady(fillSlot) := op1RespCount + 1.U
    op1RespCount := op1RespCount + 1.U
  }
  when(op2Enq) {
    op2Rows(fillSlot)(op2RespCount(rowIndexWidth - 1, 0)) := io.bankReadResp(1).bits.data
    op2RowsReady(fillSlot) := op2RespCount + 1.U
    op2RespCount := op2RespCount + 1.U
  }

  val op1FillDone = !fillNeedOp1 || op1RespCount === fillRowCount
  val op2FillDone = !fillNeedOp2 || op2RespCount === fillRowCount
  when(fillActive && op1FillDone && op2FillDone) {
    slotBankDone(fillSlot) := true.B
    when(slotFirstSent(fillSlot)) {
      slotOccupied(fillSlot) := false.B
    }
    fillSlot      := fillSlot ^ 1.U
    op1ReqRow     := 0.U
    op2ReqRow     := 0.U
    op1RespCount  := 0.U
    op2RespCount  := 0.U
  }

  for (slot <- 0 until slotCount) {
    when(slotOccupied(slot) && slotBankDone(slot) && slotFirstSent(slot)) {
      slotOccupied(slot) := false.B
    }
  }

  when(sendActive && (currentReqKind === SystolicCtrlLoadReqKind.READ_AB ||
    currentReqKind === SystolicCtrlLoadReqKind.READ_A_B_PE)) {
    assert(op1Fire === op2Fire, "SystolicArrayLoad: paired operands must fire atomically")
  }
  for (slot <- 0 until slotCount) {
    assert(op1RowsReady(slot) <= tile.U, "SystolicArrayLoad: A Buffer overflow")
    assert(op2RowsReady(slot) <= tile.U, "SystolicArrayLoad: B Buffer overflow")
    when(slotOccupied(slot) && slotBankDone(slot)) {
      assert(!needOp1(slotReq(slot).req_kind) || op1RowsReady(slot) === slotReq(slot).row_count)
      assert(!needOp2(slotReq(slot).req_kind) || op2RowsReady(slot) === slotReq(slot).row_count)
    }
  }
}
