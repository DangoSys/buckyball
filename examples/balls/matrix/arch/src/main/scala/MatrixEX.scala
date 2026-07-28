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
