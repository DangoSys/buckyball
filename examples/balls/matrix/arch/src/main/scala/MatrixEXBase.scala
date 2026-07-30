package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
abstract class SystolicArrayEXBase(val b: GlobalConfig) extends Module {
  protected val tile             = SystolicArrayConst.Tile
  protected val opElemBits       = SystolicArrayConst.OpElemBits
  protected val accElemBits      = SystolicArrayConst.AccElemBits
  protected val wsPsumBits       = 2 * opElemBits + log2Ceil(tile)
  protected val opRowBits        = SystolicArrayConst.OpRowBits
  protected val contextCount     = SystolicArrayConst.wsReuseTiles(b)
  protected val operandSlotCount = 3
  protected val contextWidth     = log2Ceil(contextCount)
  protected val operandSlotWidth = log2Ceil(operandSlotCount)
  protected val rowIndexWidth    = log2Ceil(tile)
  protected val progressWidth    = 13

  @public
  val io = IO(new Bundle {
    val load_ex_req_kind    = Input(UInt(2.W))
    val load_ex_k_tile_kind = Input(UInt(2.W))
    val load_ex_acc_slot    = Input(UInt(contextWidth.W))
    val load_ex_valid_m     = Input(UInt(5.W))
    val load_ex_valid_n     = Input(UInt(5.W))
    val load_ex_valid_k     = Input(UInt(5.W))
    val load_ex_b_valid_n   = Input(UInt(5.W))
    val load_ex_b_valid_k   = Input(UInt(5.W))
    val load_ex_weight_generation = Input(Bool())
    val load_ex_op1_i       = Flipped(Decoupled(UInt(opRowBits.W)))
    val load_ex_op2_i       = Flipped(Decoupled(UInt(opRowBits.W)))

    val ex_st_o = Decoupled(new SystolicResultRow)

  })

  protected def needOp2(kind: UInt): Bool =
    kind === SystolicCtrlLoadReqKind.READ_AB || kind === SystolicCtrlLoadReqKind.READ_A_B_PE ||
      kind === SystolicCtrlLoadReqKind.READ_A_B_BUF

  protected def isWsKind(kind: UInt): Bool =
    kind === SystolicCtrlLoadReqKind.READ_A_ONLY || kind === SystolicCtrlLoadReqKind.READ_A_B_PE ||
      kind === SystolicCtrlLoadReqKind.READ_A_B_BUF

  protected def isNewOutputTile(kTileKind: UInt): Bool =
    kTileKind === SystolicKTileKind.DIRECT || kTileKind === SystolicKTileKind.FIRST

  protected def isContinuationTile(kTileKind: UInt): Bool =
    kTileKind === SystolicKTileKind.MIDDLE || kTileKind === SystolicKTileKind.LAST

  protected def rowByte(row: UInt, idx: Int): UInt =
    row((idx + 1) * opElemBits - 1, idx * opElemBits)

  protected def dynamicRowByte(row: UInt, idx: UInt): UInt =
    (row >> (idx * opElemBits.U))(opElemBits - 1, 0)

  protected def resultRowBitsFrom(row: Vec[UInt]): UInt =
    Cat((0 until tile).reverse.map(index => row(index)))

  val contextAllocated = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val contextPendingStart = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val contextActive = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val contextFinalSeen = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val contextValidM = RegInit(VecInit(Seq.fill(contextCount)(tile.U(5.W))))
  val contextValidN = RegInit(VecInit(Seq.fill(contextCount)(tile.U(5.W))))
  val contextTotalK = RegInit(VecInit(Seq.fill(contextCount)(0.U(progressWidth.W))))
  val contextAge = RegInit(VecInit(Seq.fill(contextCount)(0.U(progressWidth.W))))
  val contextRowsComplete = RegInit(VecInit(Seq.fill(contextCount)(0.U(5.W))))
  val contextSendRow = RegInit(VecInit(Seq.fill(contextCount)(0.U(5.W))))
  val contextActiveSlot = RegInit(VecInit(Seq.fill(contextCount)(0.U(operandSlotWidth.W))))
  val contextWsMode = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val contextWeightGeneration = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val contextPendingWeightGeneration = RegInit(VecInit(Seq.fill(contextCount)(false.B)))
  val wsContextMap = RegInit(VecInit(Seq.tabulate(contextCount)(_.U(contextWidth.W))))
  val wsContextMapValid = RegInit(VecInit(Seq.fill(contextCount)(false.B)))

  val cAcc = RegInit(VecInit(Seq.tabulate(contextCount)(_ =>
    VecInit(Seq.tabulate(tile)(_ =>
      VecInit(Seq.fill(tile)(0.U(accElemBits.W))))))))

  val slotOccupied = RegInit(VecInit(Seq.fill(operandSlotCount)(false.B)))
  val slotInputComplete = RegInit(VecInit(Seq.fill(operandSlotCount)(false.B)))
  val slotUseDone = RegInit(VecInit(Seq.fill(operandSlotCount)(false.B)))
  val slotContext = RegInit(VecInit(Seq.fill(operandSlotCount)(0.U(contextWidth.W))))
  val slotReqKind = RegInit(VecInit(Seq.fill(operandSlotCount)(SystolicCtrlLoadReqKind.READ_AB)))
  val slotValidM = RegInit(VecInit(Seq.fill(operandSlotCount)(tile.U(5.W))))
  val slotValidN = RegInit(VecInit(Seq.fill(operandSlotCount)(tile.U(5.W))))
  val slotValidK = RegInit(VecInit(Seq.fill(operandSlotCount)(tile.U(5.W))))
  val slotBValidK = RegInit(VecInit(Seq.fill(operandSlotCount)(tile.U(5.W))))
  val slotARowsReceived = RegInit(VecInit(Seq.fill(operandSlotCount)(0.U(5.W))))
  val slotBRowsReceived = RegInit(VecInit(Seq.fill(operandSlotCount)(0.U(5.W))))
  val slotKTileKind = RegInit(VecInit(Seq.fill(operandSlotCount)(SystolicKTileKind.DIRECT)))
  val slotKBase = RegInit(VecInit(Seq.fill(operandSlotCount)(0.U(progressWidth.W))))
  val slotWeightGeneration = RegInit(VecInit(Seq.fill(operandSlotCount)(false.B)))
  val aRowBuf = Reg(Vec(operandSlotCount, Vec(tile, UInt(opRowBits.W))))
  val bRowBuf = Reg(Vec(operandSlotCount, Vec(tile, UInt(opRowBits.W))))
  val aRowValid = RegInit(VecInit(Seq.tabulate(operandSlotCount)(_ =>
    VecInit(Seq.fill(tile)(false.B)))))
  val bRowValid = RegInit(VecInit(Seq.tabulate(operandSlotCount)(_ =>
    VecInit(Seq.fill(tile)(false.B)))))

  val bPipeData = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(0.U(opElemBits.W))))))
  val bPipeValid = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(false.B)))))
  val bPipeContext = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(0.U(contextWidth.W))))))

  val wsBBuffer = Reg(Vec(tile, UInt(opRowBits.W)))
  val wsWeightBankValid = RegInit(VecInit(Seq.fill(2)(false.B)))
  val wsWeightBankValidN = RegInit(VecInit(Seq.fill(2)(tile.U(5.W))))
  val wsWeightBankValidK = RegInit(VecInit(Seq.fill(2)(tile.U(5.W))))
  val wsFinalReusePending = RegInit(false.B)
  val wsPrefetchGeneration = RegInit(false.B)

  val chainValid = RegInit(false.B)
  val chainContext = RegInit(0.U(contextWidth.W))

  val receiveActive = RegInit(false.B)
  val receiveSlot = RegInit(0.U(operandSlotWidth.W))
  val receiveWeightGeneration = RegInit(false.B)
  val receiveBValidN = RegInit(tile.U(5.W))

  val outputOrder = Module(new Queue(UInt(contextWidth.W), contextCount))
  val segmentOrder = Module(new Queue(UInt((contextWidth + operandSlotWidth).W), operandSlotCount))

  val hasFreeSlot = !slotOccupied.asUInt.andR
  val freeSlot = PriorityEncoder(~slotOccupied.asUInt)
  val hasFreeContext = !contextAllocated.asUInt.andR
  val freeContext = PriorityEncoder(~contextAllocated.asUInt)
  val anyContextActive = contextActive.asUInt.orR
  val activeContext = PriorityEncoder(contextActive.asUInt)
  val pipelineAdvance = WireDefault(false.B)
}
