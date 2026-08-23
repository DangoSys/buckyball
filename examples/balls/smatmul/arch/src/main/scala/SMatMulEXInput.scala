package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig

abstract class SystolicArrayEXInput(b: GlobalConfig) extends SystolicArrayEXBase(b) {
  // Load 保证同一 tile 的两条行流不会交错。EX 在第一条 A 或 B 真正握手时占用
  // receiveSlot，之后 A/B 使用各自行号写入同一个 slot，直到两条流都完成。
  val firstReceiveRow          = !receiveActive
  val activeReqKind            =
    Mux(firstReceiveRow, io.load_ex_req_kind, slotReqKind(receiveSlot))
  val activeValidM             =
    Mux(firstReceiveRow, io.load_ex_valid_m, slotValidM(receiveSlot))
  val activeBValidN            = Mux(firstReceiveRow, io.load_ex_b_valid_n, receiveBValidN)
  val activeBValidK            =
    Mux(firstReceiveRow, io.load_ex_b_valid_k, slotBValidK(receiveSlot))
  val activeARowLimit          = activeValidM
  val activeBRowLimit          = Mux(needOp2(activeReqKind), activeBValidK, 0.U(5.W))
  val activeARowsReceived      =
    Mux(firstReceiveRow, 0.U(5.W), slotARowsReceived(receiveSlot))
  val activeBRowsReceived      =
    Mux(firstReceiveRow, 0.U(5.W), slotBRowsReceived(receiveSlot))
  val firstLoadsPeWeights      =
    io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_A_B_PE
  val firstPrefetchesWeights   =
    io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_A_B_BUF
  val firstIsNewOutput         = isNewOutputTile(io.load_ex_k_tile_kind)
  val firstIsContinuation      = isContinuationTile(io.load_ex_k_tile_kind)
  val firstUsesExplicitContext = isWsKind(io.load_ex_req_kind)
  val firstWeightGeneration    = io.load_ex_weight_generation
  val firstWritesWeightBank    = firstLoadsPeWeights || firstPrefetchesWeights
  val firstBWeightGeneration   =
    Mux(firstPrefetchesWeights, !firstWeightGeneration, firstWeightGeneration)
  val firstWsMappedContext     = wsContextMap(io.load_ex_acc_slot)

  val firstWsMappingValid = wsContextMapValid(io.load_ex_acc_slot) &&
    contextAllocated(firstWsMappedContext) && contextWsMode(
      firstWsMappedContext
    )

  val firstTargetContext = Mux(
    firstUsesExplicitContext,
    Mux(firstIsContinuation, firstWsMappedContext, freeContext),
    Mux(firstIsContinuation, chainContext, freeContext)
  )

  val firstNeedsLaunchQueue             =
    firstIsNewOutput || (firstIsContinuation && firstUsesExplicitContext)
  val firstNewContextResourcesAvailable = hasFreeContext &&
    (firstUsesExplicitContext || hasFreeOsAccBank)

  val firstContextAvailable = Mux(
    firstIsContinuation,
    Mux(
      firstUsesExplicitContext,
      firstWsMappingValid && !contextPendingStart(firstTargetContext),
      contextAllocated(firstTargetContext) && chainValid
    ),
    firstNewContextResourcesAvailable
  )

  def weightBankInUse(generation: Bool): Bool = (0 until contextCount)
    .map { context =>
      contextWsMode(context) && ((contextActive(
        context
      ) && contextWeightGeneration(context) === generation) ||
        (contextPendingStart(context) && contextPendingWeightGeneration(
          context
        ) === generation))
    }
    .reduce(_ || _)

  val firstWeightBankInUse = weightBankInUse(firstBWeightGeneration)

  val anyOsContextReserved = (0 until contextCount)
    .map { context =>
      (contextActive(context) || contextPendingStart(
        context
      )) && !contextWsMode(context)
    }
    .reduce(_ || _)

  val firstModeSafe = !firstWritesWeightBank ||
    (!anyOsContextReserved && (!firstWeightBankInUse || firstPrefetchesWeights))

  val firstUsesResidentWeights =
    io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_A_ONLY ||
      firstPrefetchesWeights

  val firstResidentBankSafe = firstWeightGeneration || !anyOsContextReserved

  val firstResidentWeightsMatch =
    wsWeightBankValid(firstWeightGeneration.asUInt) &&
      wsWeightBankValidN(firstWeightGeneration.asUInt) === io.load_ex_valid_n &&
      wsWeightBankValidK(firstWeightGeneration.asUInt) === io.load_ex_valid_k &&
      firstResidentBankSafe

  val firstBaseCanReceive = hasFreeSlot &&
    firstContextAvailable && (!firstNeedsLaunchQueue || segmentOrder.io.enq.ready) &&
    (!firstIsNewOutput || outputOrder.io.enq.ready) &&
    firstModeSafe &&
    (!firstUsesResidentWeights || firstResidentWeightsMatch)

  val firstPeRowsSafe         = true.B
  val firstCanReceive         =
    firstBaseCanReceive && (!firstLoadsPeWeights || firstPeRowsSafe)
  val activeNeedOp2           = needOp2(activeReqKind)
  val activeLoadsPeWeights    =
    activeReqKind === SystolicCtrlLoadReqKind.READ_A_B_PE
  val activePrefetchesWeights =
    activeReqKind === SystolicCtrlLoadReqKind.READ_A_B_BUF
  val activeWeightGeneration  =
    Mux(firstReceiveRow, firstWeightGeneration, receiveWeightGeneration)

  val activeBWeightGeneration = Mux(
    activePrefetchesWeights,
    !activeWeightGeneration,
    activeWeightGeneration
  )

  val activeWeightBankWriteSafe =
    (activeLoadsPeWeights || !weightBankInUse(activeBWeightGeneration)) &&
      (activeBWeightGeneration || !anyOsContextReserved)

  val activePeRowSafe         = true.B
  val receiveAllowed          = Mux(firstReceiveRow, firstCanReceive, true.B)
  val activeARowIndex         = activeARowsReceived(rowIndexWidth - 1, 0)
  val activeBRowIndex         = activeBRowsReceived(rowIndexWidth - 1, 0)
  val currentSlot             = Mux(firstReceiveRow, freeSlot, receiveSlot)
  val activeARowOverwriteSafe = !aRowValid(currentSlot)(activeARowIndex)
  val activeBRowOverwriteSafe = !bRowValid(currentSlot)(activeBRowIndex)

  io.load_ex_op1_i.ready := receiveAllowed &&
    activeARowsReceived < activeARowLimit && activeARowOverwriteSafe
  io.load_ex_op2_i.ready := receiveAllowed && activeNeedOp2 &&
    activeBRowsReceived < activeBRowLimit &&
    Mux(
      activeLoadsPeWeights || activePrefetchesWeights,
      activePeRowSafe && activeWeightBankWriteSafe,
      activeBRowOverwriteSafe
    )

  val op1Fire            = io.load_ex_op1_i.fire
  val op2Fire            = io.load_ex_op2_i.fire
  val receiveEvent       = op1Fire || op2Fire
  val firstReceiveEvent  = firstReceiveRow && receiveEvent
  val currentARowIndex   = activeARowIndex
  val currentBRowIndex   = activeBRowIndex
  val aFinishesThisCycle =
    op1Fire && activeARowsReceived + 1.U >= activeARowLimit
  val bFinishesThisCycle =
    op2Fire && activeBRowsReceived + 1.U >= activeBRowLimit
  val aDoneNext          = activeARowsReceived === activeARowLimit || aFinishesThisCycle
  val bDoneNext          =
    !activeNeedOp2 || activeBRowsReceived === activeBRowLimit || bFinishesThisCycle
  val inputCompleteNext  = aDoneNext && bDoneNext
  outputOrder.io.enq.valid  := firstReceiveEvent && firstIsNewOutput
  outputOrder.io.enq.bits   := firstTargetContext
  segmentOrder.io.enq.valid := firstReceiveEvent && firstNeedsLaunchQueue
  segmentOrder.io.enq.bits  := Cat(firstTargetContext, freeSlot)

  when(op1Fire) {
    aRowBuf(currentSlot)(currentARowIndex)   := io.load_ex_op1_i.bits
    aRowValid(currentSlot)(currentARowIndex) := true.B
  }

  when(op2Fire) {
    when(activeLoadsPeWeights || activePrefetchesWeights) {
      when(activeBWeightGeneration) {
        wsBBuffer(currentBRowIndex) := io.load_ex_op2_i.bits
      }.otherwise {
        for (col <- 0 until tile) {
          bPipeData(currentBRowIndex)(col) := rowByte(
            io.load_ex_op2_i.bits,
            col
          )
        }
      }
    }.otherwise {
      bRowBuf(currentSlot)(currentBRowIndex)   := io.load_ex_op2_i.bits
      bRowValid(currentSlot)(currentBRowIndex) := true.B
    }
  }

  when(firstReceiveEvent) {
    slotOccupied(freeSlot)         := true.B
    slotInputComplete(freeSlot)    := inputCompleteNext
    slotUseDone(freeSlot)          := false.B
    slotLaunchPending(freeSlot)    := firstNeedsLaunchQueue
    slotContext(freeSlot)          := firstTargetContext
    slotReqKind(freeSlot)          := io.load_ex_req_kind
    slotValidM(freeSlot)           := io.load_ex_valid_m
    slotValidN(freeSlot)           := io.load_ex_valid_n
    slotValidK(freeSlot)           := io.load_ex_valid_k
    slotBValidK(freeSlot)          := io.load_ex_b_valid_k
    slotARowsReceived(freeSlot)    := Mux(op1Fire, 1.U, 0.U)
    slotBRowsReceived(freeSlot)    := Mux(op2Fire, 1.U, 0.U)
    slotKTileKind(freeSlot)        := io.load_ex_k_tile_kind
    slotKBase(freeSlot)            := Mux(
      firstUsesExplicitContext || firstIsNewOutput,
      0.U,
      contextTotalK(firstTargetContext)
    )
    slotWeightGeneration(freeSlot) := firstWeightGeneration
    when(firstUsesExplicitContext) {
      contextPendingWeightGeneration(
        firstTargetContext
      ) := firstWeightGeneration
    }
    receiveWeightGeneration        := firstWeightGeneration
    receiveBValidN                 := io.load_ex_b_valid_n
    when(firstNeedsLaunchQueue) {
      contextPendingStart(firstTargetContext) := true.B
    }

    when(firstIsNewOutput) {
      when(firstUsesExplicitContext) {
        for (logicalSlot <- 0 until contextCount) {
          when(
            wsContextMapValid(logicalSlot) &&
              wsContextMap(logicalSlot) === firstTargetContext
          ) {
            wsContextMapValid(logicalSlot) := false.B
          }
        }
        wsContextMap(io.load_ex_acc_slot) := firstTargetContext
        wsContextMapValid(io.load_ex_acc_slot) := true.B
      }
      contextAllocated(firstTargetContext)        := true.B
      contextFinalSeen(firstTargetContext)        :=
        io.load_ex_k_tile_kind === SystolicKTileKind.DIRECT
      contextValidM(firstTargetContext)           := io.load_ex_valid_m
      contextValidN(firstTargetContext)           := io.load_ex_valid_n
      contextTotalK(firstTargetContext)           := Mux(
        firstUsesExplicitContext,
        0.U,
        io.load_ex_valid_k.pad(progressWidth)
      )
      contextAge(firstTargetContext)              := 0.U
      contextRowsComplete(firstTargetContext)     := 0.U
      contextSendRow(firstTargetContext)          := 0.U
      contextWsMode(firstTargetContext)           := firstUsesExplicitContext
      contextWsSegmentPending(firstTargetContext) := false.B
      contextWsRowsCommitted(firstTargetContext)  := 0.U

      when(!firstUsesExplicitContext) {
        contextOsAccBank(firstTargetContext) := freeOsAccBank
        osAccBankAllocated(freeOsAccBank)    := true.B
        osAccBankOwner(freeOsAccBank)        := firstTargetContext
        for (bank <- 0 until osAccBankCount) {
          when(freeOsAccBank === bank.U) {
            for (row <- 0 until tile) {
              for (col <- 0 until tile) {
                osAcc(bank)(row)(col) := 0.U
              }
            }
          }
        }
      }

      when(
        io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_AB &&
          io.load_ex_k_tile_kind === SystolicKTileKind.FIRST
      ) {
        chainValid   := true.B
        chainContext := firstTargetContext
      }
    }.otherwise {
      assert(
        firstIsContinuation,
        "SystolicArrayEX: invalid continuation request"
      )
      assert(
        contextWsMode(firstTargetContext) === firstUsesExplicitContext,
        "SystolicArrayEX: dataflow mode changed inside a K-tile chain"
      )
      assert(
        contextValidM(firstTargetContext) === io.load_ex_valid_m,
        "SystolicArrayEX: M extent changed inside a K-tile chain"
      )
      assert(
        contextValidN(firstTargetContext) === io.load_ex_valid_n,
        "SystolicArrayEX: N extent changed inside a K-tile chain"
      )
      when(
        io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_AB &&
          io.load_ex_k_tile_kind === SystolicKTileKind.LAST
      ) {
        chainValid := false.B
      }
      when(!firstUsesExplicitContext) {
        contextTotalK(firstTargetContext) :=
          contextTotalK(firstTargetContext) + io.load_ex_valid_k.pad(
            progressWidth
          )
        when(io.load_ex_k_tile_kind === SystolicKTileKind.LAST) {
          contextFinalSeen(firstTargetContext) := true.B
        }
      }
    }

    when(firstUsesResidentWeights) {
      assert(
        wsWeightBankValid(firstWeightGeneration.asUInt),
        "SystolicArrayEX: WS request arrived without resident weights"
      )
      assert(
        wsWeightBankValidN(
          firstWeightGeneration.asUInt
        ) === io.load_ex_valid_n &&
          wsWeightBankValidK(
            firstWeightGeneration.asUInt
          ) === io.load_ex_valid_k,
        "SystolicArrayEX: WS request metadata does not match PE weights"
      )
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
    assert(
      !wsFinalReusePending,
      "SystolicArrayEX: started a new WS prefetch before the final old-weight reuse arrived"
    )
    wsFinalReusePending := true.B
  }.elsewhen(firstReceiveEvent && wsFinalReusePending) {
    assert(
      io.load_ex_req_kind === SystolicCtrlLoadReqKind.READ_A_ONLY &&
        firstWeightGeneration =/= wsPrefetchGeneration,
      "SystolicArrayEX: WS prefetch was not followed by the final old-weight reuse"
    )
    wsFinalReusePending := false.B
  }

  when(firstReceiveEvent) {
    receiveSlot   := freeSlot
    receiveActive := !inputCompleteNext
  }.elsewhen(!firstReceiveRow && receiveEvent) {
    when(inputCompleteNext) {
      slotInputComplete(receiveSlot) := true.B
    }
    receiveActive := !inputCompleteNext
  }

  when(
    op2Fire && activeBRowsReceived === 0.U &&
      (activeLoadsPeWeights || activePrefetchesWeights)
  ) {
    wsWeightBankValid(activeBWeightGeneration.asUInt)  := false.B
    wsWeightBankValidN(activeBWeightGeneration.asUInt) := activeBValidN
    wsWeightBankValidK(activeBWeightGeneration.asUInt) := activeBValidK
  }

  when(
    receiveEvent && inputCompleteNext &&
      (activeLoadsPeWeights || activePrefetchesWeights)
  ) {
    wsWeightBankValid(activeBWeightGeneration.asUInt)  := true.B
    wsWeightBankValidN(activeBWeightGeneration.asUInt) := activeBValidN
    wsWeightBankValidK(activeBWeightGeneration.asUInt) := activeBValidK
  }
}
