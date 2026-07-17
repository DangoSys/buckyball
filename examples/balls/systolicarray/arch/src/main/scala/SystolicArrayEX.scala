package examples.balls.systolicarray

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class SystolicArrayEX(val b: GlobalConfig) extends Module {
  private val tile             = SystolicArrayConst.Tile
  private val opElemBits       = SystolicArrayConst.OpElemBits
  private val accElemBits      = SystolicArrayConst.AccElemBits
  private val opRowBits        = SystolicArrayConst.OpRowBits
  private val contextCount     = SystolicArrayConst.WsReuseTiles
  private val operandSlotCount = 2
  private val contextWidth     = log2Ceil(contextCount)
  private val operandSlotWidth = log2Ceil(operandSlotCount)
  private val rowIndexWidth    = log2Ceil(tile)
  private val progressWidth    = 13

  @public
  val io = IO(new Bundle {
    val load_ex_req_kind    = Input(UInt(2.W))
    val load_ex_k_tile_kind = Input(UInt(2.W))
    val load_ex_acc_slot    = Input(UInt(contextWidth.W))
    val load_ex_valid_m     = Input(UInt(5.W))
    val load_ex_valid_n     = Input(UInt(5.W))
    val load_ex_valid_k     = Input(UInt(5.W))
    val load_ex_row_count   = Input(UInt(5.W))
    val load_ex_first_row   = Input(Bool())
    val load_ex_last_row    = Input(Bool())
    val load_ex_op1_i       = Flipped(Decoupled(UInt(opRowBits.W)))
    val load_ex_op2_i       = Flipped(Decoupled(UInt(opRowBits.W)))

    val ex_st_o = Decoupled(new SystolicResultRow)
  })

  private def needOp1(kind: UInt): Bool =
    kind === SystolicCtrlLoadReqKind.READ_AB || kind === SystolicCtrlLoadReqKind.READ_A_ONLY ||
      kind === SystolicCtrlLoadReqKind.READ_A_B_PE

  private def needOp2(kind: UInt): Bool =
    kind === SystolicCtrlLoadReqKind.READ_AB || kind === SystolicCtrlLoadReqKind.READ_A_B_PE

  private def isWsKind(kind: UInt): Bool =
    kind === SystolicCtrlLoadReqKind.READ_A_ONLY || kind === SystolicCtrlLoadReqKind.READ_A_B_PE

  private def isNewOutputTile(kind: UInt, kTileKind: UInt): Bool =
    (kind === SystolicCtrlLoadReqKind.READ_AB ||
      kind === SystolicCtrlLoadReqKind.READ_A_ONLY ||
      kind === SystolicCtrlLoadReqKind.READ_A_B_PE) &&
      (kTileKind === SystolicKTileKind.DIRECT || kTileKind === SystolicKTileKind.FIRST)

  private def isContinuationTile(kind: UInt, kTileKind: UInt): Bool =
    (kind === SystolicCtrlLoadReqKind.READ_AB ||
      kind === SystolicCtrlLoadReqKind.READ_A_ONLY ||
      kind === SystolicCtrlLoadReqKind.READ_A_B_PE) &&
      (kTileKind === SystolicKTileKind.MIDDLE || kTileKind === SystolicKTileKind.LAST)

  private def rowByte(row: UInt, idx: Int): UInt =
    row((idx + 1) * opElemBits - 1, idx * opElemBits)

  private def dynamicRowByte(row: UInt, idx: UInt): UInt =
    (row >> (idx * opElemBits.U))(opElemBits - 1, 0)

  private def resultRowBitsFrom(row: Vec[UInt]): UInt =
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
  val slotRowCount = RegInit(VecInit(Seq.fill(operandSlotCount)(tile.U(5.W))))
  val slotRowsReceived = RegInit(VecInit(Seq.fill(operandSlotCount)(0.U(5.W))))
  val slotKTileKind = RegInit(VecInit(Seq.fill(operandSlotCount)(SystolicKTileKind.DIRECT)))
  val slotWeightGeneration = RegInit(VecInit(Seq.fill(operandSlotCount)(false.B)))
  val slotABuf = Reg(Vec(operandSlotCount, Vec(tile, UInt(opRowBits.W))))
  val slotBBuf = Reg(Vec(operandSlotCount, Vec(tile, UInt(opRowBits.W))))

  val bPipeData = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(0.U(opElemBits.W))))))
  val bPipeValid = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(false.B)))))
  val bPipeContext = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(0.U(contextWidth.W))))))
  val bPipeWeightGeneration = RegInit(VecInit(Seq.fill(tile)(false.B)))

  val wsWeightsValid = RegInit(false.B)
  val wsWeightValidN = RegInit(tile.U(5.W))
  val wsWeightValidK = RegInit(tile.U(5.W))
  val wsActiveWeightGeneration = RegInit(false.B)

  val chainValid = RegInit(false.B)
  val chainContext = RegInit(0.U(contextWidth.W))

  val receiveActive = RegInit(false.B)
  val receiveRowIdx = RegInit(0.U(5.W))
  val receiveReqKind = RegInit(SystolicCtrlLoadReqKind.READ_AB)
  val receiveRowCount = RegInit(tile.U(5.W))
  val receiveSlot = RegInit(0.U(operandSlotWidth.W))
  val receiveWeightGeneration = RegInit(false.B)
  val receiveRetiringWeights = RegInit(false.B)
  val receiveRetireContext = RegInit(0.U(contextWidth.W))

  val outputOrder = Module(new Queue(UInt(contextWidth.W), contextCount))
  val segmentOrder = Module(new Queue(UInt((contextWidth + operandSlotWidth).W), operandSlotCount))

  val hasFreeSlot = !slotOccupied.asUInt.andR
  val freeSlot = PriorityEncoder(~slotOccupied.asUInt)
  val hasFreeContext = !contextAllocated.asUInt.andR
  val freeContext = PriorityEncoder(~contextAllocated.asUInt)
  val anyContextActive = contextActive.asUInt.orR
  val activeContext = PriorityEncoder(contextActive.asUInt)
  val pipelineAdvance = WireDefault(false.B)

  private def weightRowSafe(context: UInt, row: UInt): Bool = {
    val lastUseAge = contextValidM(context).pad(progressWidth) +
      contextValidN(context).pad(progressWidth) + row.pad(progressWidth) - 2.U
    !contextActive(context) || row.pad(progressWidth) >= contextTotalK(context) ||
      contextAge(context) > lastUseAge ||
      (pipelineAdvance && contextAge(context) === lastUseAge)
  }

  val firstReceiveRow = !receiveActive
  val activeReqKind = Mux(firstReceiveRow, io.load_ex_req_kind, receiveReqKind)
  val activeRowCount = Mux(firstReceiveRow, io.load_ex_row_count, receiveRowCount)
  val activeRowIdx = Mux(firstReceiveRow, 0.U, receiveRowIdx)
  val firstLoadsPeWeights = io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_A_B_PE
  val firstIsNewOutput = isNewOutputTile(io.load_ex_req_kind, io.load_ex_k_tile_kind)
  val firstIsContinuation = isContinuationTile(io.load_ex_req_kind, io.load_ex_k_tile_kind)
  val firstUsesExplicitContext = isWsKind(io.load_ex_req_kind)
  val firstWeightGeneration = Mux(
    firstLoadsPeWeights && wsWeightsValid,
    !wsActiveWeightGeneration,
    wsActiveWeightGeneration)
  val firstTargetContext = Mux(
    firstUsesExplicitContext,
    io.load_ex_acc_slot,
    Mux(firstIsContinuation, chainContext, freeContext))
  val firstContextAvailable = Mux(
    firstIsContinuation,
    contextAllocated(firstTargetContext) && !contextPendingStart(firstTargetContext) &&
      (firstUsesExplicitContext || chainValid),
    Mux(firstUsesExplicitContext, !contextAllocated(firstTargetContext), hasFreeContext))
  val firstBaseCanReceive = (firstIsNewOutput || firstIsContinuation) && hasFreeSlot &&
    firstContextAvailable && segmentOrder.io.enq.ready &&
    (!firstIsNewOutput || outputOrder.io.enq.ready) &&
    (io.load_ex_req_kind =/= SystolicCtrlLoadReqKind.READ_A_ONLY || wsWeightsValid)
  val firstPeRowSafe = !wsWeightsValid || (!segmentOrder.io.deq.valid &&
    (!anyContextActive || weightRowSafe(activeContext, 0.U)))
  val firstCanReceive = firstBaseCanReceive && (!firstLoadsPeWeights || firstPeRowSafe)
  val activeNeedOp1 = needOp1(activeReqKind)
  val activeNeedOp2 = needOp2(activeReqKind)
  val activeLoadsPeWeights = activeReqKind === SystolicCtrlLoadReqKind.READ_A_B_PE
  val activePeRowSafe = !receiveRetiringWeights ||
    weightRowSafe(receiveRetireContext, activeRowIdx)
  val receiveAllowed = Mux(
    firstReceiveRow,
    firstCanReceive,
    !activeLoadsPeWeights || activePeRowSafe)

  io.load_ex_op1_i.ready := receiveAllowed && activeNeedOp1 &&
    (!activeNeedOp2 || io.load_ex_op2_i.valid)
  io.load_ex_op2_i.ready := receiveAllowed && activeNeedOp2 &&
    (!activeNeedOp1 || io.load_ex_op1_i.valid)

  val op1Fire = io.load_ex_op1_i.fire
  val op2Fire = io.load_ex_op2_i.fire
  val rowDone = receiveAllowed && (op1Fire || !activeNeedOp1) &&
    (op2Fire || !activeNeedOp2) && (op1Fire || op2Fire)
  val firstRowDone = rowDone && firstReceiveRow
  val currentSlot = Mux(firstReceiveRow, freeSlot, receiveSlot)
  val currentRowIndex = activeRowIdx(rowIndexWidth - 1, 0)

  outputOrder.io.enq.valid := firstRowDone && firstIsNewOutput
  outputOrder.io.enq.bits := firstTargetContext
  segmentOrder.io.enq.valid := firstRowDone
  segmentOrder.io.enq.bits := Cat(firstTargetContext, freeSlot)

  when(op1Fire) {
    slotABuf(currentSlot)(currentRowIndex) := io.load_ex_op1_i.bits
  }

  when(op2Fire) {
    when(activeReqKind === SystolicCtrlLoadReqKind.READ_A_B_PE) {
      for (col <- 0 until tile) {
        bPipeData(currentRowIndex)(col) := rowByte(io.load_ex_op2_i.bits, col)
        bPipeValid(currentRowIndex)(col) :=
          activeRowIdx < io.load_ex_valid_k && col.U < io.load_ex_valid_n
      }
      bPipeWeightGeneration(currentRowIndex) := Mux(
        firstReceiveRow,
        firstWeightGeneration,
        receiveWeightGeneration)
    }.otherwise {
      slotBBuf(currentSlot)(currentRowIndex) := io.load_ex_op2_i.bits
    }
  }

  when(firstRowDone) {
    slotOccupied(freeSlot) := true.B
    slotInputComplete(freeSlot) := io.load_ex_last_row
    slotUseDone(freeSlot) := false.B
    slotContext(freeSlot) := firstTargetContext
    slotReqKind(freeSlot) := io.load_ex_req_kind
    slotValidM(freeSlot) := io.load_ex_valid_m
    slotValidN(freeSlot) := io.load_ex_valid_n
    slotValidK(freeSlot) := io.load_ex_valid_k
    slotRowCount(freeSlot) := io.load_ex_row_count
    slotRowsReceived(freeSlot) := 1.U
    slotKTileKind(freeSlot) := io.load_ex_k_tile_kind
    slotWeightGeneration(freeSlot) := firstWeightGeneration
    contextPendingStart(firstTargetContext) := true.B

    when(firstIsNewOutput) {
      contextAllocated(firstTargetContext) := true.B
      contextFinalSeen(firstTargetContext) := false.B
      contextValidM(firstTargetContext) := io.load_ex_valid_m
      contextValidN(firstTargetContext) := io.load_ex_valid_n
      contextTotalK(firstTargetContext) := 0.U
      contextAge(firstTargetContext) := 0.U
      contextRowsComplete(firstTargetContext) := 0.U
      contextSendRow(firstTargetContext) := 0.U

      for (context <- 0 until contextCount) {
        when(firstTargetContext === context.U) {
          for (row <- 0 until tile) {
            for (col <- 0 until tile) {
              cAcc(context)(row)(col) := 0.U
            }
          }
        }
      }

      when(io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_AB &&
        io.load_ex_k_tile_kind === SystolicKTileKind.FIRST) {
        chainValid := true.B
        chainContext := firstTargetContext
      }
    }.otherwise {
      assert(firstIsContinuation, "SystolicArrayEX: invalid continuation request")
      assert(contextWsMode(firstTargetContext) === firstUsesExplicitContext,
        "SystolicArrayEX: dataflow mode changed inside a K-tile chain")
      assert(contextValidM(firstTargetContext) === io.load_ex_valid_m,
        "SystolicArrayEX: M extent changed inside a K-tile chain")
      assert(contextValidN(firstTargetContext) === io.load_ex_valid_n,
        "SystolicArrayEX: N extent changed inside a K-tile chain")
      when(io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_AB &&
        io.load_ex_k_tile_kind === SystolicKTileKind.LAST) {
        chainValid := false.B
      }
    }

    when(io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_A_ONLY) {
      assert(wsWeightsValid, "SystolicArrayEX: READ_A_ONLY arrived without PE weights")
      assert(wsWeightValidN === io.load_ex_valid_n && wsWeightValidK === io.load_ex_valid_k,
        "SystolicArrayEX: READ_A_ONLY metadata does not match PE weights")
    }

    when(firstLoadsPeWeights) {
      receiveWeightGeneration := firstWeightGeneration
      receiveRetiringWeights := wsWeightsValid && anyContextActive
      receiveRetireContext := activeContext
    }
  }

  when(rowDone && !firstReceiveRow) {
    slotRowsReceived(receiveSlot) := slotRowsReceived(receiveSlot) + 1.U
    when(io.load_ex_last_row) {
      slotInputComplete(receiveSlot) := true.B
    }
  }

  when(rowDone) {
    assert(io.load_ex_first_row === firstReceiveRow,
      "SystolicArrayEX: invalid first-row marker")
    assert(io.load_ex_last_row === (activeRowIdx === (activeRowCount - 1.U)),
      "SystolicArrayEX: invalid last-row marker")

    when(io.load_ex_last_row) {
      receiveActive := false.B
      receiveRowIdx := 0.U
      when(activeReqKind === SystolicCtrlLoadReqKind.READ_A_B_PE) {
        wsWeightsValid := true.B
        wsWeightValidN := io.load_ex_valid_n
        wsWeightValidK := io.load_ex_valid_k
        wsActiveWeightGeneration := receiveWeightGeneration
        receiveRetiringWeights := false.B
      }
    }.otherwise {
      receiveActive := true.B
      receiveRowIdx := activeRowIdx + 1.U
      when(firstReceiveRow) {
        receiveReqKind := io.load_ex_req_kind
        receiveRowCount := io.load_ex_row_count
        receiveSlot := freeSlot
      }
    }
  }

  val contextInputsReady = Wire(Vec(contextCount, Bool()))
  val aInjectValid = Wire(Vec(tile, Bool()))
  val aInjectData = Wire(Vec(tile, UInt(opElemBits.W)))
  val aInjectContext = Wire(Vec(tile, UInt(contextWidth.W)))
  val aInjectMRow = Wire(Vec(tile, UInt(5.W)))
  val bInjectValid = Wire(Vec(tile, Bool()))
  val bInjectData = Wire(Vec(tile, UInt(opElemBits.W)))
  val bInjectContext = Wire(Vec(tile, UInt(contextWidth.W)))

  for (context <- 0 until contextCount) {
    contextInputsReady(context) := true.B
  }

  for (row <- 0 until tile) {
    val contextHit = Wire(Vec(contextCount, Bool()))
    val contextData = Wire(Vec(contextCount, UInt(opElemBits.W)))
    val contextMRow = Wire(Vec(contextCount, UInt(5.W)))

    for (context <- 0 until contextCount) {
      val logicalIndex = contextAge(context) - row.U
      val expectsOsA = contextActive(context) && !contextWsMode(context) &&
        row.U < contextValidM(context) && contextAge(context) >= row.U &&
        logicalIndex < contextTotalK(context)
      val expectsWsA = contextActive(context) && contextWsMode(context) &&
        row.U < contextTotalK(context) && contextAge(context) >= row.U &&
        logicalIndex < contextValidM(context)
      val expectsA = expectsOsA || expectsWsA
      val sourceFound = WireDefault(false.B)
      val sourceReady = WireDefault(false.B)
      val sourceData = WireDefault(0.U(opElemBits.W))
      val weightsReady = bPipeWeightGeneration(row) === contextWeightGeneration(context) &&
        (0 until tile).map(col =>
          col.U >= contextValidN(context) || bPipeValid(row)(col)).reduce(_ && _)

      for (slot <- 0 until operandSlotCount) {
        val matches = slotOccupied(slot) && slotContext(slot) === context.U &&
          contextActiveSlot(context) === slot.U
        when(matches) {
          sourceFound := true.B
          when(contextWsMode(context)) {
            sourceReady := slotRowsReceived(slot) > logicalIndex && weightsReady
            sourceData := rowByte(
              slotABuf(slot)(logicalIndex(rowIndexWidth - 1, 0)),
              row)
          }.otherwise {
            sourceReady := slotRowsReceived(slot) > row.U
            sourceData := dynamicRowByte(slotABuf(slot)(row), logicalIndex)
          }
        }
      }

      when(expectsA && !(sourceFound && sourceReady)) {
        contextInputsReady(context) := false.B
      }
      contextHit(context) := expectsA && sourceFound && sourceReady
      contextData(context) := sourceData
      contextMRow(context) := Mux(contextWsMode(context), logicalIndex(4, 0), row.U)
    }

    assert(PopCount(contextHit) <= 1.U,
      "SystolicArrayEX: multiple contexts attempted to inject into one A row")
    aInjectValid(row) := contextHit.asUInt.orR
    aInjectData(row) := Mux1H(contextHit, contextData)
    aInjectContext(row) := PriorityEncoder(contextHit.asUInt)
    aInjectMRow(row) := Mux1H(contextHit, contextMRow)
  }

  for (col <- 0 until tile) {
    val contextHit = Wire(Vec(contextCount, Bool()))
    val contextData = Wire(Vec(contextCount, UInt(opElemBits.W)))

    for (context <- 0 until contextCount) {
      val logicalK = contextAge(context) - col.U
      val expectsB = contextActive(context) && !contextWsMode(context) &&
        col.U < contextValidN(context) &&
        contextAge(context) >= col.U && logicalK < contextTotalK(context)
      val sourceFound = WireDefault(false.B)
      val sourceReady = WireDefault(false.B)
      val sourceData = WireDefault(0.U(opElemBits.W))

      for (slot <- 0 until operandSlotCount) {
        val matches = slotOccupied(slot) && slotContext(slot) === context.U &&
          contextActiveSlot(context) === slot.U
        val bRow = slotBBuf(slot)(logicalK(rowIndexWidth - 1, 0))
        val bReady = slotRowsReceived(slot) > logicalK
        when(matches) {
          sourceFound := true.B
          sourceReady := bReady
          sourceData := rowByte(bRow, col)
        }
      }

      when(expectsB && !(sourceFound && sourceReady)) {
        contextInputsReady(context) := false.B
      }
      contextHit(context) := expectsB && sourceFound && sourceReady
      contextData(context) := sourceData
    }

    assert(PopCount(contextHit) <= 1.U,
      "SystolicArrayEX: multiple contexts attempted to inject into one B column")
    bInjectValid(col) := contextHit.asUInt.orR
    bInjectData(col) := Mux1H(contextHit, contextData)
    bInjectContext(col) := PriorityEncoder(contextHit.asUInt)
  }

  pipelineAdvance := anyContextActive &&
    (0 until contextCount).map(context =>
      !contextActive(context) || contextInputsReady(context)).reduce(_ && _)

  val aPipeData = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(0.U(opElemBits.W))))))
  val aPipeValid = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(false.B)))))
  val aPipeContext = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(0.U(contextWidth.W))))))
  val aPipeMRow = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(0.U(5.W))))))
  val aStepData = Wire(Vec(tile, Vec(tile, UInt(opElemBits.W))))
  val aStepValid = Wire(Vec(tile, Vec(tile, Bool())))
  val aStepContext = Wire(Vec(tile, Vec(tile, UInt(contextWidth.W))))
  val aStepMRow = Wire(Vec(tile, Vec(tile, UInt(5.W))))
  val bStepData = Wire(Vec(tile, Vec(tile, UInt(opElemBits.W))))
  val bStepValid = Wire(Vec(tile, Vec(tile, Bool())))
  val bStepContext = Wire(Vec(tile, Vec(tile, UInt(contextWidth.W))))
  val wsPsumData = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(0.U(accElemBits.W))))))
  val wsPsumValid = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(false.B)))))
  val wsPsumContext = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(0.U(contextWidth.W))))))
  val wsPsumMRow = RegInit(VecInit(Seq.tabulate(tile)(_ =>
    VecInit(Seq.fill(tile)(0.U(5.W))))))

  for (row <- 0 until tile) {
    for (col <- 0 until tile) {
      if (col == 0) {
        aStepData(row)(col) := aInjectData(row)
        aStepValid(row)(col) := aInjectValid(row)
        aStepContext(row)(col) := aInjectContext(row)
        aStepMRow(row)(col) := aInjectMRow(row)
      } else {
        val sourceContext = aPipeContext(row)(col - 1)
        val sourceValidN = contextValidN(sourceContext)
        aStepData(row)(col) := aPipeData(row)(col - 1)
        aStepValid(row)(col) := aPipeValid(row)(col - 1) && col.U < sourceValidN
        aStepContext(row)(col) := sourceContext
        aStepMRow(row)(col) := aPipeMRow(row)(col - 1)
      }

      if (row == 0) {
        bStepData(row)(col) := bInjectData(col)
        bStepValid(row)(col) := bInjectValid(col)
        bStepContext(row)(col) := bInjectContext(col)
      } else {
        val sourceContext = bPipeContext(row - 1)(col)
        val sourceValidM = contextValidM(sourceContext)
        bStepData(row)(col) := bPipeData(row - 1)(col)
        bStepValid(row)(col) := bPipeValid(row - 1)(col) && row.U < sourceValidM
        bStepContext(row)(col) := sourceContext
      }
    }
  }

  when(pipelineAdvance) {
    for (row <- 0 until tile) {
      for (col <- 0 until tile) {
        aPipeData(row)(col) := aStepData(row)(col)
        aPipeValid(row)(col) := aStepValid(row)(col)
        aPipeContext(row)(col) := aStepContext(row)(col)
        aPipeMRow(row)(col) := aStepMRow(row)(col)

        when(!contextWsMode(activeContext)) {
          bPipeData(row)(col) := bStepData(row)(col)
          bPipeValid(row)(col) := bStepValid(row)(col)
          bPipeContext(row)(col) := bStepContext(row)(col)
        }

        wsPsumValid(row)(col) := false.B

        when(aStepValid(row)(col) && contextWsMode(aStepContext(row)(col))) {
          val targetContext = aStepContext(row)(col)
          val product = aStepData(row)(col).asSInt * bPipeData(row)(col).asSInt
          val partialSum = Wire(UInt(accElemBits.W))

          assert(bPipeValid(row)(col), "SystolicArrayEX: WS used an invalid PE weight")
          assert(bPipeWeightGeneration(row) === contextWeightGeneration(targetContext),
            "SystolicArrayEX: WS used the wrong PE weight generation")

          if (row == 0) {
            partialSum := product.pad(accElemBits).asUInt
          } else {
            assert(wsPsumValid(row - 1)(col),
              "SystolicArrayEX: WS partial sum did not arrive from the previous PE row")
            assert(wsPsumContext(row - 1)(col) === targetContext,
              "SystolicArrayEX: WS partial-sum context changed between PE rows")
            assert(wsPsumMRow(row - 1)(col) === aStepMRow(row)(col),
              "SystolicArrayEX: WS partial-sum M row changed between PE rows")
            partialSum := (wsPsumData(row - 1)(col).asSInt +
              product.pad(accElemBits)).asUInt
          }

          wsPsumData(row)(col) := partialSum
          wsPsumValid(row)(col) := true.B
          wsPsumContext(row)(col) := targetContext
          wsPsumMRow(row)(col) := aStepMRow(row)(col)

          when((row + 1).U === contextTotalK(targetContext)) {
            cAcc(targetContext)(aStepMRow(row)(col)(rowIndexWidth - 1, 0))(col) :=
              cAcc(targetContext)(aStepMRow(row)(col)(rowIndexWidth - 1, 0))(col) + partialSum
          }
        }.elsewhen(aStepValid(row)(col) && bStepValid(row)(col)) {
          assert(aStepContext(row)(col) === bStepContext(row)(col),
            "SystolicArrayEX: A/B context tags do not match")
          val product = aStepData(row)(col).asSInt * bStepData(row)(col).asSInt
          val targetContext = aStepContext(row)(col)
          cAcc(targetContext)(row)(col) :=
            cAcc(targetContext)(row)(col) + product.pad(accElemBits).asUInt
        }
      }
    }

    for (context <- 0 until contextCount) {
      when(contextActive(context)) {
        val firstCompleteCycle = contextValidN(context).pad(progressWidth) +
          contextTotalK(context) - 2.U
        val lastCycle = contextValidM(context).pad(progressWidth) +
          contextValidN(context).pad(progressWidth) + contextTotalK(context) - 3.U

        when(contextFinalSeen(context) && contextAge(context) >= firstCompleteCycle &&
          contextRowsComplete(context) < contextValidM(context)) {
          contextRowsComplete(context) := contextRowsComplete(context) + 1.U
        }

        when(contextAge(context) >= lastCycle) {
          contextActive(context) := false.B
          contextAge(context) := 0.U
        }.otherwise {
          contextAge(context) := contextAge(context) + 1.U
        }
      }
    }

  }

  for (slot <- 0 until operandSlotCount) {
    val slotContextAge = contextAge(slotContext(slot))
    val slotContextActive = contextActive(slotContext(slot)) &&
      contextActiveSlot(slotContext(slot)) === slot.U
    val maxExtent = Mux(slotValidM(slot) >= slotValidN(slot),
      slotValidM(slot), slotValidN(slot))
    val osLastUseCycle = slotValidK(slot) - 1.U + maxExtent - 1.U
    val wsLastUseCycle = slotValidK(slot) - 1.U + slotValidM(slot) - 1.U
    val lastUseCycle = Mux(isWsKind(slotReqKind(slot)), wsLastUseCycle, osLastUseCycle)
    val reachedLastUse = pipelineAdvance && slotContextActive && slotContextAge >= lastUseCycle

    when(slotOccupied(slot) && reachedLastUse) {
      slotUseDone(slot) := true.B
    }
    when(slotOccupied(slot) && slotInputComplete(slot) && (slotUseDone(slot) || reachedLastUse)) {
      slotOccupied(slot) := false.B
      slotInputComplete(slot) := false.B
      slotUseDone(slot) := false.B
      slotRowsReceived(slot) := 0.U
    }
  }

  val segmentContext = segmentOrder.io.deq.bits(
    contextWidth + operandSlotWidth - 1, operandSlotWidth)
  val segmentSlot = segmentOrder.io.deq.bits(operandSlotWidth - 1, 0)
  val launchSegment = segmentOrder.io.deq.valid && !anyContextActive &&
    slotOccupied(segmentSlot) && contextPendingStart(segmentContext)
  segmentOrder.io.deq.ready := launchSegment

  when(launchSegment) {
    assert(slotContext(segmentSlot) === segmentContext,
      "SystolicArrayEX: segment queue metadata does not match its operand slot")
    contextPendingStart(segmentContext) := false.B
    contextActive(segmentContext) := true.B
    contextActiveSlot(segmentContext) := segmentSlot
    contextWsMode(segmentContext) := isWsKind(slotReqKind(segmentSlot))
    contextWeightGeneration(segmentContext) := slotWeightGeneration(segmentSlot)
    contextAge(segmentContext) := 0.U
    contextTotalK(segmentContext) := slotValidK(segmentSlot).pad(progressWidth)
    contextFinalSeen(segmentContext) :=
      slotKTileKind(segmentSlot) === SystolicKTileKind.DIRECT ||
        slotKTileKind(segmentSlot) === SystolicKTileKind.LAST

    for (row <- 0 until tile) {
      for (col <- 0 until tile) {
        aPipeValid(row)(col) := false.B
        wsPsumValid(row)(col) := false.B
        when(!isWsKind(slotReqKind(segmentSlot))) {
          bPipeValid(row)(col) := false.B
        }
      }
    }
  }

  val outputContext = outputOrder.io.deq.bits
  val outputRowsComplete = contextRowsComplete(outputContext)
  val outputSendRow = contextSendRow(outputContext)
  val outputValidM = contextValidM(outputContext)
  val outputResult = resultRowBitsFrom(
    cAcc(outputContext)(outputSendRow(rowIndexWidth - 1, 0)))

  io.ex_st_o.valid := outputOrder.io.deq.valid && outputRowsComplete > outputSendRow
  io.ex_st_o.bits.data := outputResult

  val finishingOutput = io.ex_st_o.fire && outputSendRow + 1.U >= outputValidM
  outputOrder.io.deq.ready := finishingOutput

  when(io.ex_st_o.fire) {
    when(finishingOutput) {
      assert(!contextActive(outputContext),
        "SystolicArrayEX: context released before its final MAC completed")
      contextAllocated(outputContext) := false.B
      contextFinalSeen(outputContext) := false.B
      contextTotalK(outputContext) := 0.U
      contextRowsComplete(outputContext) := 0.U
      contextSendRow(outputContext) := 0.U
    }.otherwise {
      contextSendRow(outputContext) := outputSendRow + 1.U
    }
  }

  when(rowDone && firstReceiveRow) {
    assert(io.load_ex_valid_m >= 1.U && io.load_ex_valid_m <= tile.U)
    assert(io.load_ex_valid_n >= 1.U && io.load_ex_valid_n <= tile.U)
    assert(io.load_ex_valid_k >= 1.U && io.load_ex_valid_k <= tile.U)
    assert(io.load_ex_row_count >= 1.U && io.load_ex_row_count <= tile.U)
    assert(firstIsNewOutput || firstIsContinuation,
      "SystolicArrayEX: invalid output segment kind")
    when(firstLoadsPeWeights) {
      assert(io.load_ex_row_count === tile.U,
        "SystolicArrayEX: READ_A_B_PE must replace every PE weight row")
    }
    when(firstIsContinuation && io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_AB) {
      assert(chainValid, "SystolicArrayEX: continuation arrived without an active K-tile chain")
    }
  }

  when(receiveActive) {
    assert(receiveRowIdx < receiveRowCount,
      "SystolicArrayEX: receive row index overflow")
  }
  for (context <- 0 until contextCount) {
    assert(contextRowsComplete(context) <= contextValidM(context),
      "SystolicArrayEX: completed row count overflow")
    when(contextActive(context) || contextPendingStart(context)) {
      assert(contextAllocated(context),
        "SystolicArrayEX: active context is not allocated")
    }
  }
  for (slot <- 0 until operandSlotCount) {
    assert(slotRowsReceived(slot) <= slotRowCount(slot),
      "SystolicArrayEX: operand slot row count overflow")
    when(slotOccupied(slot) && !slotUseDone(slot)) {
      assert(contextAllocated(slotContext(slot)),
        "SystolicArrayEX: operand slot refers to a free context")
    }
    when(slotUseDone(slot)) {
      assert(slotOccupied(slot),
        "SystolicArrayEX: retired operand slot is not occupied")
    }
  }
  when(outputOrder.io.deq.valid) {
    assert(contextAllocated(outputOrder.io.deq.bits),
      "SystolicArrayEX: output queue refers to a free context")
  }
}
