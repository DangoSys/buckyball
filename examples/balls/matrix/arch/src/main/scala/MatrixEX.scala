package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class SystolicArrayEX(val b: GlobalConfig) extends Module {
  private val tile             = SystolicArrayConst.Tile
  private val opElemBits       = SystolicArrayConst.OpElemBits
  private val accElemBits      = SystolicArrayConst.AccElemBits
  private val wsPsumBits       = 2 * opElemBits + log2Ceil(tile)
  private val opRowBits        = SystolicArrayConst.OpRowBits
  private val contextCount     = SystolicArrayConst.WsReuseTiles
  private val operandSlotCount = 3
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
    val load_ex_b_valid_n   = Input(UInt(5.W))
    val load_ex_b_valid_k   = Input(UInt(5.W))
    val load_ex_weight_generation = Input(Bool())
    val load_ex_op1_i       = Flipped(Decoupled(UInt(opRowBits.W)))
    val load_ex_op2_i       = Flipped(Decoupled(UInt(opRowBits.W)))

    val ex_st_o = Decoupled(new SystolicResultRow)

  })

  private def needOp2(kind: UInt): Bool =
    kind === SystolicCtrlLoadReqKind.READ_AB || kind === SystolicCtrlLoadReqKind.READ_A_B_PE ||
      kind === SystolicCtrlLoadReqKind.READ_A_B_BUF

  private def isWsKind(kind: UInt): Bool =
    kind === SystolicCtrlLoadReqKind.READ_A_ONLY || kind === SystolicCtrlLoadReqKind.READ_A_B_PE ||
      kind === SystolicCtrlLoadReqKind.READ_A_B_BUF

  private def isNewOutputTile(kTileKind: UInt): Bool =
    kTileKind === SystolicKTileKind.DIRECT || kTileKind === SystolicKTileKind.FIRST

  private def isContinuationTile(kTileKind: UInt): Bool =
    kTileKind === SystolicKTileKind.MIDDLE || kTileKind === SystolicKTileKind.LAST

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

  // Load 保证同一 tile 的两条行流不会交错。EX 在第一条 A 或 B 真正握手时占用
  // receiveSlot，之后 A/B 使用各自行号写入同一个 slot，直到两条流都完成。
  val firstReceiveRow = !receiveActive
  val activeReqKind = Mux(firstReceiveRow, io.load_ex_req_kind, slotReqKind(receiveSlot))
  val activeValidM = Mux(firstReceiveRow, io.load_ex_valid_m, slotValidM(receiveSlot))
  val activeBValidN = Mux(firstReceiveRow, io.load_ex_b_valid_n, receiveBValidN)
  val activeBValidK = Mux(firstReceiveRow, io.load_ex_b_valid_k, slotBValidK(receiveSlot))
  val activeARowLimit = activeValidM
  val activeBRowLimit = Mux(needOp2(activeReqKind), activeBValidK, 0.U(5.W))
  val activeARowsReceived = Mux(firstReceiveRow, 0.U(5.W), slotARowsReceived(receiveSlot))
  val activeBRowsReceived = Mux(firstReceiveRow, 0.U(5.W), slotBRowsReceived(receiveSlot))
  val firstLoadsPeWeights = io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_A_B_PE
  val firstPrefetchesWeights = io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_A_B_BUF
  val firstIsNewOutput = isNewOutputTile(io.load_ex_k_tile_kind)
  val firstIsContinuation = isContinuationTile(io.load_ex_k_tile_kind)
  val firstUsesExplicitContext = isWsKind(io.load_ex_req_kind)
  val firstWeightGeneration = io.load_ex_weight_generation
  val firstWritesWeightBank = firstLoadsPeWeights || firstPrefetchesWeights
  val firstBWeightGeneration = Mux(firstPrefetchesWeights,
    !firstWeightGeneration, firstWeightGeneration)
  val firstWsMappedContext = wsContextMap(io.load_ex_acc_slot)
  val firstWsMappingValid = wsContextMapValid(io.load_ex_acc_slot) &&
    contextAllocated(firstWsMappedContext) && contextWsMode(firstWsMappedContext)
  val firstTargetContext = Mux(
    firstUsesExplicitContext,
    Mux(firstIsContinuation, firstWsMappedContext, freeContext),
    Mux(firstIsContinuation, chainContext, freeContext))
  val firstNeedsLaunchQueue = firstIsNewOutput || (firstIsContinuation && firstUsesExplicitContext)
  val firstContextAvailable = Mux(
    firstIsContinuation,
    Mux(firstUsesExplicitContext,
      firstWsMappingValid && !contextPendingStart(firstTargetContext),
      contextAllocated(firstTargetContext) && chainValid),
    hasFreeContext)
  def weightBankInUse(generation: Bool): Bool = (0 until contextCount).map { context =>
    contextWsMode(context) && (
      (contextActive(context) && contextWeightGeneration(context) === generation) ||
      (contextPendingStart(context) && contextPendingWeightGeneration(context) === generation))
  }.reduce(_ || _)
  val firstWeightBankInUse = weightBankInUse(firstBWeightGeneration)
  val anyOsContextReserved = (0 until contextCount).map { context =>
    (contextActive(context) || contextPendingStart(context)) && !contextWsMode(context)
  }.reduce(_ || _)
  val firstModeSafe = !firstWritesWeightBank ||
    (!anyOsContextReserved && (!firstWeightBankInUse || firstPrefetchesWeights))
  val firstUsesResidentWeights = io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_A_ONLY ||
    firstPrefetchesWeights
  val firstResidentBankSafe = firstWeightGeneration || !anyOsContextReserved
  val firstResidentWeightsMatch = wsWeightBankValid(firstWeightGeneration.asUInt) &&
    wsWeightBankValidN(firstWeightGeneration.asUInt) === io.load_ex_valid_n &&
    wsWeightBankValidK(firstWeightGeneration.asUInt) === io.load_ex_valid_k &&
    firstResidentBankSafe
  val firstBaseCanReceive = hasFreeSlot &&
    firstContextAvailable && (!firstNeedsLaunchQueue || segmentOrder.io.enq.ready) &&
    (!firstIsNewOutput || outputOrder.io.enq.ready) &&
    firstModeSafe &&
    (!firstUsesResidentWeights || firstResidentWeightsMatch)
  val firstPeRowsSafe = true.B
  val firstCanReceive = firstBaseCanReceive && (!firstLoadsPeWeights || firstPeRowsSafe)
  val activeNeedOp2 = needOp2(activeReqKind)
  val activeLoadsPeWeights = activeReqKind === SystolicCtrlLoadReqKind.READ_A_B_PE
  val activePrefetchesWeights = activeReqKind === SystolicCtrlLoadReqKind.READ_A_B_BUF
  val activeWeightGeneration = Mux(firstReceiveRow,
    firstWeightGeneration, receiveWeightGeneration)
  val activeBWeightGeneration = Mux(activePrefetchesWeights,
    !activeWeightGeneration, activeWeightGeneration)
  val activeWeightBankWriteSafe =
    (activeLoadsPeWeights || !weightBankInUse(activeBWeightGeneration)) &&
      (activeBWeightGeneration || !anyOsContextReserved)
  val activePeRowSafe = true.B
  val receiveAllowed = Mux(firstReceiveRow, firstCanReceive, true.B)
  val activeARowIndex = activeARowsReceived(rowIndexWidth - 1, 0)
  val activeBRowIndex = activeBRowsReceived(rowIndexWidth - 1, 0)
  val currentSlot = Mux(firstReceiveRow, freeSlot, receiveSlot)
  val activeARowOverwriteSafe = !aRowValid(currentSlot)(activeARowIndex)
  val activeBRowOverwriteSafe = !bRowValid(currentSlot)(activeBRowIndex)

  io.load_ex_op1_i.ready := receiveAllowed &&
    activeARowsReceived < activeARowLimit && activeARowOverwriteSafe
  io.load_ex_op2_i.ready := receiveAllowed && activeNeedOp2 &&
    activeBRowsReceived < activeBRowLimit &&
    Mux(activeLoadsPeWeights || activePrefetchesWeights,
      activePeRowSafe && activeWeightBankWriteSafe, activeBRowOverwriteSafe)


  val op1Fire = io.load_ex_op1_i.fire
  val op2Fire = io.load_ex_op2_i.fire
  val receiveEvent = op1Fire || op2Fire
  val firstReceiveEvent = firstReceiveRow && receiveEvent
  val currentARowIndex = activeARowIndex
  val currentBRowIndex = activeBRowIndex
  val aFinishesThisCycle = op1Fire && activeARowsReceived + 1.U >= activeARowLimit
  val bFinishesThisCycle = op2Fire && activeBRowsReceived + 1.U >= activeBRowLimit
  val aDoneNext = activeARowsReceived === activeARowLimit || aFinishesThisCycle
  val bDoneNext = !activeNeedOp2 || activeBRowsReceived === activeBRowLimit || bFinishesThisCycle
  val inputCompleteNext = aDoneNext && bDoneNext
  outputOrder.io.enq.valid := firstReceiveEvent && firstIsNewOutput
  outputOrder.io.enq.bits := firstTargetContext
  segmentOrder.io.enq.valid := firstReceiveEvent && firstNeedsLaunchQueue
  segmentOrder.io.enq.bits := Cat(firstTargetContext, freeSlot)

  when(op1Fire) {
    aRowBuf(currentSlot)(currentARowIndex) := io.load_ex_op1_i.bits
    aRowValid(currentSlot)(currentARowIndex) := true.B
  }

  when(op2Fire) {
    when(activeLoadsPeWeights || activePrefetchesWeights) {
      when(activeBWeightGeneration) {
        wsBBuffer(currentBRowIndex) := io.load_ex_op2_i.bits
      }.otherwise {
        for (col <- 0 until tile) {
          bPipeData(currentBRowIndex)(col) := rowByte(io.load_ex_op2_i.bits, col)
        }
      }
    }.otherwise {
      bRowBuf(currentSlot)(currentBRowIndex) := io.load_ex_op2_i.bits
      bRowValid(currentSlot)(currentBRowIndex) := true.B
    }
  }

  when(firstReceiveEvent) {
    slotOccupied(freeSlot) := true.B
    slotInputComplete(freeSlot) := inputCompleteNext
    slotUseDone(freeSlot) := false.B
    slotContext(freeSlot) := firstTargetContext
    slotReqKind(freeSlot) := io.load_ex_req_kind
    slotValidM(freeSlot) := io.load_ex_valid_m
    slotValidN(freeSlot) := io.load_ex_valid_n
    slotValidK(freeSlot) := io.load_ex_valid_k
    slotBValidK(freeSlot) := io.load_ex_b_valid_k
    slotARowsReceived(freeSlot) := Mux(op1Fire, 1.U, 0.U)
    slotBRowsReceived(freeSlot) := Mux(op2Fire, 1.U, 0.U)
    slotKTileKind(freeSlot) := io.load_ex_k_tile_kind
    slotKBase(freeSlot) := Mux(
      firstUsesExplicitContext || firstIsNewOutput,
      0.U,
      contextTotalK(firstTargetContext))
    slotWeightGeneration(freeSlot) := firstWeightGeneration
    when(firstUsesExplicitContext) {
      contextPendingWeightGeneration(firstTargetContext) := firstWeightGeneration
    }
    receiveWeightGeneration := firstWeightGeneration
    receiveBValidN := io.load_ex_b_valid_n
    when(firstNeedsLaunchQueue) {
      contextPendingStart(firstTargetContext) := true.B
    }

    when(firstIsNewOutput) {
      when(firstUsesExplicitContext) {
        for (logicalSlot <- 0 until contextCount) {
          when(wsContextMapValid(logicalSlot) &&
            wsContextMap(logicalSlot) === firstTargetContext) {
            wsContextMapValid(logicalSlot) := false.B
          }
        }
        wsContextMap(io.load_ex_acc_slot) := firstTargetContext
        wsContextMapValid(io.load_ex_acc_slot) := true.B
      }
      contextAllocated(firstTargetContext) := true.B
      contextFinalSeen(firstTargetContext) :=
        io.load_ex_k_tile_kind === SystolicKTileKind.DIRECT
      contextValidM(firstTargetContext) := io.load_ex_valid_m
      contextValidN(firstTargetContext) := io.load_ex_valid_n
      contextTotalK(firstTargetContext) := Mux(
        firstUsesExplicitContext,
        0.U,
        io.load_ex_valid_k.pad(progressWidth))
      contextAge(firstTargetContext) := 0.U
      contextRowsComplete(firstTargetContext) := 0.U
      contextSendRow(firstTargetContext) := 0.U
      contextWsMode(firstTargetContext) := firstUsesExplicitContext

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
      when(!firstUsesExplicitContext) {
        contextTotalK(firstTargetContext) :=
          contextTotalK(firstTargetContext) + io.load_ex_valid_k.pad(progressWidth)
        when(io.load_ex_k_tile_kind === SystolicKTileKind.LAST) {
          contextFinalSeen(firstTargetContext) := true.B
        }
      }
    }

    when(firstUsesResidentWeights) {
      assert(wsWeightBankValid(firstWeightGeneration.asUInt),
        "SystolicArrayEX: WS request arrived without resident weights")
      assert(wsWeightBankValidN(firstWeightGeneration.asUInt) === io.load_ex_valid_n &&
        wsWeightBankValidK(firstWeightGeneration.asUInt) === io.load_ex_valid_k,
        "SystolicArrayEX: WS request metadata does not match PE weights")
    }
    when(firstPrefetchesWeights) {
      wsPrefetchGeneration := !firstWeightGeneration
    }
  }

  when(!firstReceiveRow && op1Fire) {
    slotARowsReceived(receiveSlot) := slotARowsReceived(receiveSlot) + 1.U
  }
  when(!firstReceiveRow && op2Fire) {
    slotBRowsReceived(receiveSlot) := slotBRowsReceived(receiveSlot) + 1.U
  }

  when(firstReceiveEvent && firstPrefetchesWeights) {
    assert(!wsFinalReusePending,
      "SystolicArrayEX: started a new WS prefetch before the final old-weight reuse arrived")
    wsFinalReusePending := true.B
  }.elsewhen(firstReceiveEvent && wsFinalReusePending) {
    assert(io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_A_ONLY &&
      firstWeightGeneration =/= wsPrefetchGeneration,
      "SystolicArrayEX: WS prefetch was not followed by the final old-weight reuse")
    wsFinalReusePending := false.B
  }

  when(firstReceiveEvent) {
    receiveSlot := freeSlot
    receiveActive := !inputCompleteNext
  }.elsewhen(!firstReceiveRow && receiveEvent) {
    when(inputCompleteNext) {
      slotInputComplete(receiveSlot) := true.B
    }
    receiveActive := !inputCompleteNext
  }

  when(op2Fire && activeBRowsReceived === 0.U &&
    (activeLoadsPeWeights || activePrefetchesWeights)) {
    wsWeightBankValid(activeBWeightGeneration.asUInt) := false.B
    wsWeightBankValidN(activeBWeightGeneration.asUInt) := activeBValidN
    wsWeightBankValidK(activeBWeightGeneration.asUInt) := activeBValidK
  }

  when(receiveEvent && inputCompleteNext &&
    (activeLoadsPeWeights || activePrefetchesWeights)) {
    wsWeightBankValid(activeBWeightGeneration.asUInt) := true.B
    wsWeightBankValidN(activeBWeightGeneration.asUInt) := activeBValidN
    wsWeightBankValidK(activeBWeightGeneration.asUInt) := activeBValidK
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
      val waitsForOsSegment = contextActive(context) && !contextWsMode(context) &&
        !contextFinalSeen(context) && row.U < contextValidM(context) &&
        contextAge(context) >= row.U && logicalIndex >= contextTotalK(context)
      val sourceFound = WireDefault(false.B)
      val sourceReady = WireDefault(false.B)
      val sourceData = WireDefault(0.U(opElemBits.W))
      val sourceRowsReceived = WireDefault(false.B)
      val sourceRowValid = WireDefault(false.B)
      val sourceOwnerMatches = WireDefault(false.B)
      val sourceWeightsReady = WireDefault(true.B)
      val contextWeightBank = contextWeightGeneration(context).asUInt
      val weightsReady = wsWeightBankValid(contextWeightBank) &&
        wsWeightBankValidN(contextWeightBank) === contextValidN(context) &&
        wsWeightBankValidK(contextWeightBank).pad(progressWidth) === contextTotalK(context)

      for (slot <- 0 until operandSlotCount) {
        val slotEndK = slotKBase(slot) + slotValidK(slot).pad(progressWidth)
        val osMatches = logicalIndex >= slotKBase(slot) && logicalIndex < slotEndK
        val wsMatches = contextActiveSlot(context) === slot.U
        val matches = slotOccupied(slot) && slotContext(slot) === context.U &&
          Mux(contextWsMode(context), wsMatches, osMatches)
        when(matches) {
          sourceFound := true.B
          when(contextWsMode(context)) {
            val aRowIndex = logicalIndex(rowIndexWidth - 1, 0)
            sourceRowsReceived := slotARowsReceived(slot) > logicalIndex
            sourceRowValid := aRowValid(slot)(aRowIndex)
            sourceOwnerMatches := true.B
            sourceWeightsReady := weightsReady
            sourceReady := sourceRowsReceived && sourceRowValid &&
              sourceOwnerMatches && sourceWeightsReady
            sourceData := rowByte(
              aRowBuf(slot)(aRowIndex),
              row)
          }.otherwise {
            val localK = logicalIndex - slotKBase(slot)
            sourceRowsReceived := slotARowsReceived(slot) > row.U
            sourceRowValid := aRowValid(slot)(row)
            sourceOwnerMatches := true.B
            sourceReady := sourceRowsReceived && sourceRowValid && sourceOwnerMatches
            sourceData := dynamicRowByte(aRowBuf(slot)(row), localK)
          }
        }
      }

      when(waitsForOsSegment || (expectsA && !(sourceFound && sourceReady))) {
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
      val waitsForOsSegment = contextActive(context) && !contextWsMode(context) &&
        !contextFinalSeen(context) && col.U < contextValidN(context) &&
        contextAge(context) >= col.U && logicalK >= contextTotalK(context)
      val sourceFound = WireDefault(false.B)
      val sourceReady = WireDefault(false.B)
      val sourceData = WireDefault(0.U(opElemBits.W))
      val sourceRowsReceived = WireDefault(false.B)
      val sourceRowValid = WireDefault(false.B)
      val sourceOwnerMatches = WireDefault(false.B)

      for (slot <- 0 until operandSlotCount) {
        val slotEndK = slotKBase(slot) + slotValidK(slot).pad(progressWidth)
        val localK = logicalK - slotKBase(slot)
        val matches = slotOccupied(slot) && slotContext(slot) === context.U &&
          logicalK >= slotKBase(slot) && logicalK < slotEndK
        val bRowIndex = localK(rowIndexWidth - 1, 0)
        val bRow = bRowBuf(slot)(bRowIndex)
        when(matches) {
          sourceFound := true.B
          sourceRowsReceived := slotBRowsReceived(slot) > localK
          sourceRowValid := bRowValid(slot)(bRowIndex)
          sourceOwnerMatches := true.B
          sourceReady := sourceRowsReceived && sourceRowValid && sourceOwnerMatches
          sourceData := rowByte(bRow, col)
        }
      }

      when(waitsForOsSegment || (expectsB && !(sourceFound && sourceReady))) {
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
    VecInit(Seq.fill(tile)(0.U(wsPsumBits.W))))))
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
          val targetWeightBank = contextWeightGeneration(targetContext).asUInt
          val weightData = Mux(contextWeightGeneration(targetContext),
            rowByte(wsBBuffer(row), col), bPipeData(row)(col))
          val product = aStepData(row)(col).asSInt * weightData.asSInt
          val partialSum = Wire(UInt(wsPsumBits.W))

          assert(wsWeightBankValid(targetWeightBank),
            "SystolicArrayEX: WS used an invalid weight bank")
          assert(wsWeightBankValidN(targetWeightBank) === contextValidN(targetContext) &&
            wsWeightBankValidK(targetWeightBank).pad(progressWidth) ===
              contextTotalK(targetContext),
            "SystolicArrayEX: WS weight bank metadata does not match its context")

          if (row == 0) {
            partialSum := product.pad(wsPsumBits).asUInt
          } else {
            assert(wsPsumValid(row - 1)(col),
              "SystolicArrayEX: WS partial sum did not arrive from the previous PE row")
            assert(wsPsumContext(row - 1)(col) === targetContext,
              "SystolicArrayEX: WS partial-sum context changed between PE rows")
            assert(wsPsumMRow(row - 1)(col) === aStepMRow(row)(col),
              "SystolicArrayEX: WS partial-sum M row changed between PE rows")
            partialSum := (wsPsumData(row - 1)(col).asSInt +
              product.pad(wsPsumBits)).asUInt
          }

          wsPsumData(row)(col) := partialSum
          wsPsumValid(row)(col) := true.B
          wsPsumContext(row)(col) := targetContext
          wsPsumMRow(row)(col) := aStepMRow(row)(col)

          when((row + 1).U === contextTotalK(targetContext)) {
            cAcc(targetContext)(aStepMRow(row)(col)(rowIndexWidth - 1, 0))(col) :=
              cAcc(targetContext)(aStepMRow(row)(col)(rowIndexWidth - 1, 0))(col) +
                partialSum.asSInt.pad(accElemBits).asUInt
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

  val slotLaunching = Wire(Vec(operandSlotCount, Bool()))
  for (slot <- 0 until operandSlotCount) {
    val slotContextAge = contextAge(slotContext(slot))
    val slotIsWs = isWsKind(slotReqKind(slot))
    val slotContextActive = contextActive(slotContext(slot)) &&
      (!slotIsWs || contextActiveSlot(slotContext(slot)) === slot.U)
    val maxExtent = Mux(slotValidM(slot) >= slotValidN(slot),
      slotValidM(slot), slotValidN(slot))
    val osLastUseCycle = slotKBase(slot) + slotValidK(slot).pad(progressWidth) - 1.U +
      maxExtent.pad(progressWidth) - 1.U
    val wsLastUseCycle = slotValidK(slot) - 1.U + slotValidM(slot) - 1.U
    val lastUseCycle = Mux(slotIsWs, wsLastUseCycle, osLastUseCycle)
    val reachedLastUse = pipelineAdvance && slotContextActive && slotContextAge >= lastUseCycle

    when(slotOccupied(slot) && reachedLastUse) {
      slotUseDone(slot) := true.B
    }
    when(slotOccupied(slot) && slotInputComplete(slot) &&
      (slotUseDone(slot) || reachedLastUse) && !slotLaunching(slot)) {
      slotOccupied(slot) := false.B
      slotInputComplete(slot) := false.B
      slotUseDone(slot) := false.B
      slotARowsReceived(slot) := 0.U
      slotBRowsReceived(slot) := 0.U
      for (row <- 0 until tile) {
        aRowValid(slot)(row) := false.B
        bRowValid(slot)(row) := false.B
      }
    }
  }

  val segmentContext = segmentOrder.io.deq.bits(
    contextWidth + operandSlotWidth - 1, operandSlotWidth)
  val segmentSlot = segmentOrder.io.deq.bits(operandSlotWidth - 1, 0)
  val segmentIsWs = isWsKind(slotReqKind(segmentSlot))
  val activeContextCount = PopCount(contextActive)
  val activeOsInputsDrained = (0 until contextCount).map { context =>
    !contextActive(context) || (!contextWsMode(context) && contextFinalSeen(context) &&
      contextAge(context) + 1.U >= contextTotalK(context))
  }.reduce(_ && _)
  val canOverlapOs = anyContextActive && !segmentIsWs && activeContextCount < 2.U &&
    activeOsInputsDrained && pipelineAdvance
  val canOverlapWs = anyContextActive &&
    isWsKind(slotReqKind(segmentSlot)) && pipelineAdvance &&
    (0 until contextCount).map { context =>
      !contextActive(context) ||
        (contextWsMode(context) &&
          contextAge(context) + 1.U >= contextValidM(context).pad(progressWidth))
    }.reduce(_ && _)
  val launchAllowed = !anyContextActive || canOverlapOs || canOverlapWs
  val segmentWeightBank = slotWeightGeneration(segmentSlot).asUInt
  val segmentResidentWeightsReady = wsWeightBankValid(segmentWeightBank) &&
    wsWeightBankValidN(segmentWeightBank) === slotValidN(segmentSlot) &&
    wsWeightBankValidK(segmentWeightBank) === slotValidK(segmentSlot)
  val segmentPrefetchLaunchReady =
    slotReqKind(segmentSlot) === SystolicCtrlLoadReqKind.READ_A_B_BUF &&
      slotARowsReceived(segmentSlot) >= slotValidM(segmentSlot) &&
      segmentResidentWeightsReady
  val segmentOsStreamLaunchReady = !segmentIsWs &&
    slotARowsReceived(segmentSlot) =/= 0.U &&
    slotBRowsReceived(segmentSlot) =/= 0.U
  val segmentInputReady = slotInputComplete(segmentSlot) ||
    segmentPrefetchLaunchReady || segmentOsStreamLaunchReady
  val launchSegment = segmentOrder.io.deq.valid && launchAllowed &&
    slotOccupied(segmentSlot) && segmentInputReady &&
    !contextActive(segmentContext) &&
    contextPendingStart(segmentContext)
  segmentOrder.io.deq.ready := launchSegment
  slotLaunching := VecInit((0 until operandSlotCount).map(slot =>
    launchSegment && segmentSlot === slot.U))

  when(launchSegment) {
    assert(slotContext(segmentSlot) === segmentContext,
      "SystolicArrayEX: segment queue metadata does not match its operand slot")
    slotUseDone(segmentSlot) := false.B
    contextPendingStart(segmentContext) := false.B
    contextActive(segmentContext) := true.B
    contextActiveSlot(segmentContext) := segmentSlot
    contextWsMode(segmentContext) := isWsKind(slotReqKind(segmentSlot))
    contextWeightGeneration(segmentContext) := slotWeightGeneration(segmentSlot)
    contextAge(segmentContext) := 0.U
    when(segmentIsWs) {
      val launchWeightBank = slotWeightGeneration(segmentSlot).asUInt
      assert(wsWeightBankValid(launchWeightBank),
        "SystolicArrayEX: WS launched without a valid weight bank")
      assert(wsWeightBankValidN(launchWeightBank) === slotValidN(segmentSlot) &&
        wsWeightBankValidK(launchWeightBank) === slotValidK(segmentSlot),
        "SystolicArrayEX: WS launched with mismatched weight metadata")
      contextTotalK(segmentContext) := slotValidK(segmentSlot).pad(progressWidth)
      contextFinalSeen(segmentContext) :=
        slotKTileKind(segmentSlot) === SystolicKTileKind.DIRECT ||
          slotKTileKind(segmentSlot) === SystolicKTileKind.LAST
    }.otherwise {
      wsWeightBankValid(0) := false.B
    }

    when(!anyContextActive) {
      for (row <- 0 until tile) {
        for (col <- 0 until tile) {
          aPipeValid(row)(col) := false.B
          wsPsumValid(row)(col) := false.B
          when(!segmentIsWs) {
            bPipeValid(row)(col) := false.B
          }
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

  when(firstReceiveEvent) {
    assert(io.load_ex_valid_m >= 1.U && io.load_ex_valid_m <= tile.U)
    assert(io.load_ex_valid_n >= 1.U && io.load_ex_valid_n <= tile.U)
    assert(io.load_ex_valid_k >= 1.U && io.load_ex_valid_k <= tile.U)
    when(needOp2(io.load_ex_req_kind)) {
      assert(io.load_ex_b_valid_n >= 1.U && io.load_ex_b_valid_n <= tile.U)
      assert(io.load_ex_b_valid_k >= 1.U && io.load_ex_b_valid_k <= tile.U)
    }
    when(io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_A_B_PE) {
      assert(io.load_ex_b_valid_n === io.load_ex_valid_n &&
        io.load_ex_b_valid_k === io.load_ex_valid_k,
        "SystolicArrayEX: direct PE weights must match the current tile extent")
    }
    when(firstIsContinuation && io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_AB) {
      assert(chainValid, "SystolicArrayEX: continuation arrived without an active K-tile chain")
    }
  }

  when(receiveActive) {
    assert(slotARowsReceived(receiveSlot) <= slotValidM(receiveSlot),
      "SystolicArrayEX: A receive row index overflow")
    assert(slotBRowsReceived(receiveSlot) <=
      Mux(needOp2(slotReqKind(receiveSlot)), slotBValidK(receiveSlot), 0.U),
      "SystolicArrayEX: B receive row index overflow")
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
    assert(slotARowsReceived(slot) <= slotValidM(slot),
      "SystolicArrayEX: A operand slot row count overflow")
    assert(slotBRowsReceived(slot) <= Mux(needOp2(slotReqKind(slot)), slotBValidK(slot), 0.U),
      "SystolicArrayEX: B operand slot row count overflow")
    when(slotOccupied(slot) && slotInputComplete(slot)) {
      assert(slotARowsReceived(slot) === slotValidM(slot),
        "SystolicArrayEX: completed slot is missing A rows")
      assert(!needOp2(slotReqKind(slot)) || slotBRowsReceived(slot) === slotBValidK(slot),
        "SystolicArrayEX: completed slot is missing B rows")
    }
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
