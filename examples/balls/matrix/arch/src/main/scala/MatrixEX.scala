package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class SystolicArrayEX(b: GlobalConfig) extends SystolicArrayEXDatapath(b) {
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
      (slotUseDone(slot) || reachedLastUse) && !slotLaunchPending(slot) &&
      !slotLaunching(slot)) {
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
    (!segmentIsWs || !contextWsSegmentPending(segmentContext)) &&
    contextPendingStart(segmentContext)
  segmentOrder.io.deq.ready := launchSegment
  slotLaunching := VecInit((0 until operandSlotCount).map(slot =>
    launchSegment && segmentSlot === slot.U))

  when(launchSegment) {
    assert(slotContext(segmentSlot) === segmentContext,
      "SystolicArrayEX: segment queue metadata does not match its operand slot")
    slotUseDone(segmentSlot) := false.B
    slotLaunchPending(segmentSlot) := false.B
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
      contextWsSegmentPending(segmentContext) := true.B
      contextWsRowsCommitted(segmentContext) := 0.U
      contextWsKTileKind(segmentContext) := slotKTileKind(segmentSlot)
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
  val outputRowIndex = outputSendRow(rowIndexWidth - 1, 0)
  val outputOsAccBank = contextOsAccBank(outputContext)
  val osPackedRows = Wire(Vec(osAccBankCount, Vec(tile, UInt(accRowBits.W))))
  for (bank <- 0 until osAccBankCount) {
    for (row <- 0 until tile) {
      osPackedRows(bank)(row) := resultRowBitsFrom(osAcc(bank)(row))
    }
  }
  val osSelectedBankRows = Wire(Vec(tile, UInt(accRowBits.W)))
  for (row <- 0 until tile) {
    osSelectedBankRows(row) := Mux1H((0 until osAccBankCount).map { bank =>
      (outputOsAccBank === bank.U) -> osPackedRows(bank)(row)
    })
  }
  val osOutputResult = Mux1H((0 until tile).map { row =>
    (outputRowIndex === row.U) -> osSelectedBankRows(row)
  })
  val outputIsWs = outputOrder.io.deq.valid && contextWsMode(outputContext)

  val wsBufferedResult = wsResultBuffer.io.deq.bits
  val wsBufferedFinal = wsBufferedResult.kTileKind === SystolicKTileKind.DIRECT ||
    wsBufferedResult.kTileKind === SystolicKTileKind.LAST
  val wsBufferedMatchesOutput = outputOrder.io.deq.valid && outputIsWs &&
    wsBufferedResult.context === outputContext && wsBufferedResult.row === outputRowIndex
  val wsOutputValid = wsResultBuffer.io.deq.valid && wsBufferedFinal &&
    wsBufferedMatchesOutput
  val osOutputValid = outputOrder.io.deq.valid && !outputIsWs &&
    outputRowsComplete > outputSendRow


  io.ex_st_o.valid := Mux(outputIsWs, wsOutputValid, osOutputValid)
  io.ex_st_o.bits.data := Mux(outputIsWs, wsBufferedResult.data, osOutputResult)


  wsResultBuffer.io.deq.ready := Mux(wsBufferedFinal,
    wsBufferedMatchesOutput && io.ex_st_o.ready, true.B)
  val wsResultCommit = wsResultBuffer.io.deq.fire
  when(wsResultCommit) {
    val commitContext = wsBufferedResult.context
    val commitRow = wsBufferedResult.row
    val committedRows = contextWsRowsCommitted(commitContext)
    val committingLastRow = committedRows + 1.U >= contextValidM(commitContext)


    assert(contextAllocated(commitContext) && contextWsMode(commitContext),
      "SystolicArrayEX: WS accumulator result refers to an invalid context")
    assert(contextWsSegmentPending(commitContext),
      "SystolicArrayEX: WS accumulator result has no pending segment")
    assert(commitRow === committedRows(rowIndexWidth - 1, 0),
      "SystolicArrayEX: WS accumulator rows committed out of order")

    when(!wsBufferedFinal) {
      wsAccMem.write(wsAccAddress(commitContext, commitRow), wsBufferedResult.data)
    }.otherwise {
      contextRowsComplete(commitContext) := contextRowsComplete(commitContext) + 1.U
    }

    contextWsRowsCommitted(commitContext) := committedRows + 1.U
    when(committingLastRow) {
      contextWsSegmentPending(commitContext) := false.B
    }
  }

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
      contextWsSegmentPending(outputContext) := false.B
      contextWsRowsCommitted(outputContext) := 0.U
      when(!contextWsMode(outputContext)) {
        assert(osAccBankAllocated(outputOsAccBank),
          "SystolicArrayEX: released an unallocated OS accumulator bank")
        assert(osAccBankOwner(outputOsAccBank) === outputContext,
          "SystolicArrayEX: released an OS accumulator bank owned by another context")
        osAccBankAllocated(outputOsAccBank) := false.B
      }
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
    when(contextAllocated(context) && !contextWsMode(context)) {
      val bank = contextOsAccBank(context)
      assert(osAccBankAllocated(bank),
        "SystolicArrayEX: allocated OS context has no accumulator bank")
      assert(osAccBankOwner(bank) === context.U,
        "SystolicArrayEX: allocated OS context does not own its accumulator bank")
    }
    when(contextWsSegmentPending(context)) {
      assert(contextAllocated(context) && contextWsMode(context),
        "SystolicArrayEX: pending WS segment refers to an invalid context")
      assert(contextWsRowsCommitted(context) < contextValidM(context),
        "SystolicArrayEX: pending WS segment already committed every row")
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
