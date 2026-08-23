package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig

class SystolicWsAccumReadEntry(
  contextWidth:   Int,
  rowIndexWidth:  Int,
  accRowBits:     Int,
  partialRowBits: Int)
    extends Bundle {
  val oldRow     = UInt(accRowBits.W)
  val partialRow = UInt(partialRowBits.W)
  val context    = UInt(contextWidth.W)
  val row        = UInt(rowIndexWidth.W)
  val kTileKind  = UInt(2.W)
}

class SystolicWsAccumResultEntry(
  contextWidth:  Int,
  rowIndexWidth: Int,
  accRowBits:    Int)
    extends Bundle {
  val data      = UInt(accRowBits.W)
  val context   = UInt(contextWidth.W)
  val row       = UInt(rowIndexWidth.W)
  val kTileKind = UInt(2.W)
}

abstract class SystolicArrayEXDatapath(b: GlobalConfig) extends SystolicArrayEXInput(b) {
  val contextInputsReady = Wire(Vec(contextCount, Bool()))
  val aInjectValid       = Wire(Vec(tile, Bool()))
  val aInjectData        = Wire(Vec(tile, UInt(opElemBits.W)))
  val aInjectContext     = Wire(Vec(tile, UInt(contextWidth.W)))
  val aInjectMRow        = Wire(Vec(tile, UInt(5.W)))
  val bInjectValid       = Wire(Vec(tile, Bool()))
  val bInjectData        = Wire(Vec(tile, UInt(opElemBits.W)))
  val bInjectContext     = Wire(Vec(tile, UInt(contextWidth.W)))

  for (context <- 0 until contextCount) {
    contextInputsReady(context) := true.B
  }

  for (row <- 0 until tile) {
    val contextHit  = Wire(Vec(contextCount, Bool()))
    val contextData = Wire(Vec(contextCount, UInt(opElemBits.W)))
    val contextMRow = Wire(Vec(contextCount, UInt(5.W)))

    for (context <- 0 until contextCount) {
      val logicalIndex       = contextAge(context) - row.U
      val expectsOsA         = contextActive(context) && !contextWsMode(context) &&
        row.U < contextValidM(context) && contextAge(context) >= row.U &&
        logicalIndex < contextTotalK(context)
      val expectsWsA         = contextActive(context) && contextWsMode(context) &&
        row.U < contextTotalK(context) && contextAge(context) >= row.U &&
        logicalIndex < contextValidM(context)
      val expectsA           = expectsOsA || expectsWsA
      val waitsForOsSegment  =
        contextActive(context) && !contextWsMode(context) &&
          !contextFinalSeen(context) && row.U < contextValidM(context) &&
          contextAge(context) >= row.U && logicalIndex >= contextTotalK(context)
      val sourceFound        = WireDefault(false.B)
      val sourceReady        = WireDefault(false.B)
      val sourceData         = WireDefault(0.U(opElemBits.W))
      val sourceRowsReceived = WireDefault(false.B)
      val sourceRowValid     = WireDefault(false.B)
      val sourceOwnerMatches = WireDefault(false.B)
      val sourceWeightsReady = WireDefault(true.B)
      val contextWeightBank  = contextWeightGeneration(context).asUInt
      val weightsReady       = wsWeightBankValid(contextWeightBank) &&
        wsWeightBankValidN(contextWeightBank) === contextValidN(context) &&
        wsWeightBankValidK(contextWeightBank).pad(
          progressWidth
        ) === contextTotalK(context)

      for (slot <- 0 until operandSlotCount) {
        val slotEndK  = slotKBase(slot) + slotValidK(slot).pad(progressWidth)
        val osMatches =
          logicalIndex >= slotKBase(slot) && logicalIndex < slotEndK
        val wsMatches = contextActiveSlot(context) === slot.U
        val matches   = slotOccupied(slot) && slotContext(slot) === context.U &&
          Mux(contextWsMode(context), wsMatches, osMatches)
        when(matches) {
          sourceFound := true.B
          when(contextWsMode(context)) {
            val aRowIndex = logicalIndex(rowIndexWidth - 1, 0)
            sourceRowsReceived := slotARowsReceived(slot) > logicalIndex
            sourceRowValid     := aRowValid(slot)(aRowIndex)
            sourceOwnerMatches := true.B
            sourceWeightsReady := weightsReady
            sourceReady        := sourceRowsReceived && sourceRowValid &&
              sourceOwnerMatches && sourceWeightsReady
            sourceData         := rowByte(aRowBuf(slot)(aRowIndex), row)
          }.otherwise {
            val localK = logicalIndex - slotKBase(slot)
            sourceRowsReceived := slotARowsReceived(slot) > row.U
            sourceRowValid     := aRowValid(slot)(row)
            sourceOwnerMatches := true.B
            sourceReady        := sourceRowsReceived && sourceRowValid && sourceOwnerMatches
            sourceData         := dynamicRowByte(aRowBuf(slot)(row), localK)
          }
        }
      }

      when(waitsForOsSegment || (expectsA && !(sourceFound && sourceReady))) {
        contextInputsReady(context) := false.B
      }
      contextHit(context)  := expectsA && sourceFound && sourceReady
      contextData(context) := sourceData
      contextMRow(context) := Mux(
        contextWsMode(context),
        logicalIndex(4, 0),
        row.U
      )
    }

    assert(
      PopCount(contextHit) <= 1.U,
      "SystolicArrayEX: multiple contexts attempted to inject into one A row"
    )
    aInjectValid(row)   := contextHit.asUInt.orR
    aInjectData(row)    := Mux1H(contextHit, contextData)
    aInjectContext(row) := PriorityEncoder(contextHit.asUInt)
    aInjectMRow(row)    := Mux1H(contextHit, contextMRow)
  }

  for (col <- 0 until tile) {
    val contextHit  = Wire(Vec(contextCount, Bool()))
    val contextData = Wire(Vec(contextCount, UInt(opElemBits.W)))

    for (context <- 0 until contextCount) {
      val logicalK           = contextAge(context) - col.U
      val expectsB           = contextActive(context) && !contextWsMode(context) &&
        col.U < contextValidN(context) &&
        contextAge(context) >= col.U && logicalK < contextTotalK(context)
      val waitsForOsSegment  =
        contextActive(context) && !contextWsMode(context) &&
          !contextFinalSeen(context) && col.U < contextValidN(context) &&
          contextAge(context) >= col.U && logicalK >= contextTotalK(context)
      val sourceFound        = WireDefault(false.B)
      val sourceReady        = WireDefault(false.B)
      val sourceData         = WireDefault(0.U(opElemBits.W))
      val sourceRowsReceived = WireDefault(false.B)
      val sourceRowValid     = WireDefault(false.B)
      val sourceOwnerMatches = WireDefault(false.B)

      for (slot <- 0 until operandSlotCount) {
        val slotEndK  = slotKBase(slot) + slotValidK(slot).pad(progressWidth)
        val localK    = logicalK - slotKBase(slot)
        val matches   = slotOccupied(slot) && slotContext(slot) === context.U &&
          logicalK >= slotKBase(slot) && logicalK < slotEndK
        val bRowIndex = localK(rowIndexWidth - 1, 0)
        val bRow      = bRowBuf(slot)(bRowIndex)
        when(matches) {
          sourceFound        := true.B
          sourceRowsReceived := slotBRowsReceived(slot) > localK
          sourceRowValid     := bRowValid(slot)(bRowIndex)
          sourceOwnerMatches := true.B
          sourceReady        := sourceRowsReceived && sourceRowValid && sourceOwnerMatches
          sourceData         := rowByte(bRow, col)
        }
      }

      when(waitsForOsSegment || (expectsB && !(sourceFound && sourceReady))) {
        contextInputsReady(context) := false.B
      }
      contextHit(context)  := expectsB && sourceFound && sourceReady
      contextData(context) := sourceData
    }

    assert(
      PopCount(contextHit) <= 1.U,
      "SystolicArrayEX: multiple contexts attempted to inject into one B column"
    )
    bInjectValid(col)   := contextHit.asUInt.orR
    bInjectData(col)    := Mux1H(contextHit, contextData)
    bInjectContext(col) := PriorityEncoder(contextHit.asUInt)
  }

  val aPipeData = RegInit(
    VecInit(Seq.tabulate(tile)(_ => VecInit(Seq.fill(tile)(0.U(opElemBits.W)))))
  )

  val aPipeValid = RegInit(
    VecInit(Seq.tabulate(tile)(_ => VecInit(Seq.fill(tile)(false.B))))
  )

  val aPipeContext = RegInit(
    VecInit(
      Seq.tabulate(tile)(_ => VecInit(Seq.fill(tile)(0.U(contextWidth.W))))
    )
  )

  val aPipeMRow = RegInit(
    VecInit(Seq.tabulate(tile)(_ => VecInit(Seq.fill(tile)(0.U(5.W)))))
  )

  val aStepData    = Wire(Vec(tile, Vec(tile, UInt(opElemBits.W))))
  val aStepValid   = Wire(Vec(tile, Vec(tile, Bool())))
  val aStepContext = Wire(Vec(tile, Vec(tile, UInt(contextWidth.W))))
  val aStepMRow    = Wire(Vec(tile, Vec(tile, UInt(5.W))))
  val bStepData    = Wire(Vec(tile, Vec(tile, UInt(opElemBits.W))))
  val bStepValid   = Wire(Vec(tile, Vec(tile, Bool())))
  val bStepContext = Wire(Vec(tile, Vec(tile, UInt(contextWidth.W))))

  val wsPsumData = RegInit(
    VecInit(Seq.tabulate(tile)(_ => VecInit(Seq.fill(tile)(0.U(wsPsumBits.W)))))
  )

  val wsPsumValid = RegInit(
    VecInit(Seq.tabulate(tile)(_ => VecInit(Seq.fill(tile)(false.B))))
  )

  val wsPsumContext = RegInit(
    VecInit(
      Seq.tabulate(tile)(_ => VecInit(Seq.fill(tile)(0.U(contextWidth.W))))
    )
  )

  val wsPsumMRow = RegInit(
    VecInit(Seq.tabulate(tile)(_ => VecInit(Seq.fill(tile)(0.U(5.W)))))
  )

  for (row <- 0 until tile) {
    for (col <- 0 until tile) {
      if (col == 0) {
        aStepData(row)(col)    := aInjectData(row)
        aStepValid(row)(col)   := aInjectValid(row)
        aStepContext(row)(col) := aInjectContext(row)
        aStepMRow(row)(col)    := aInjectMRow(row)
      } else {
        val sourceContext = aPipeContext(row)(col - 1)
        val sourceValidN  = contextValidN(sourceContext)
        aStepData(row)(col)    := aPipeData(row)(col - 1)
        aStepValid(row)(col)   := aPipeValid(row)(col - 1) && col.U < sourceValidN
        aStepContext(row)(col) := sourceContext
        aStepMRow(row)(col)    := aPipeMRow(row)(col - 1)
      }

      if (row == 0) {
        bStepData(row)(col)    := bInjectData(col)
        bStepValid(row)(col)   := bInjectValid(col)
        bStepContext(row)(col) := bInjectContext(col)
      } else {
        val sourceContext = bPipeContext(row - 1)(col)
        val sourceValidM  = contextValidM(sourceContext)
        bStepData(row)(col)    := bPipeData(row - 1)(col)
        bStepValid(row)(col)   := bPipeValid(row - 1)(col) && row.U < sourceValidM
        bStepContext(row)(col) := sourceContext
      }
    }
  }

  val wsPartialSum   = Wire(Vec(tile, Vec(tile, UInt(wsPsumBits.W))))
  val wsComputeValid = Wire(Vec(tile, Vec(tile, Bool())))
  val wsBypassValid  = Wire(Vec(tile, Vec(tile, Bool())))
  val wsStageData    = Wire(Vec(tile, Vec(tile, UInt(wsPsumBits.W))))
  val wsStageValid   = Wire(Vec(tile, Vec(tile, Bool())))
  val wsStageContext = Wire(Vec(tile, Vec(tile, UInt(contextWidth.W))))
  val wsStageMRow    = Wire(Vec(tile, Vec(tile, UInt(5.W))))
  for (row <- 0 until tile) {
    for (col <- 0 until tile) {
      val targetContext = aStepContext(row)(col)
      val weightData    = Mux(
        contextWeightGeneration(targetContext),
        rowByte(wsBBuffer(row), col),
        bPipeData(row)(col)
      )
      val product       = aStepData(row)(col).asSInt * weightData.asSInt
      wsComputeValid(row)(col) := aStepValid(row)(col) && contextWsMode(
        targetContext
      )
      if (row == 0) {
        wsPartialSum(row)(col)   := product.pad(wsPsumBits).asUInt
        wsBypassValid(row)(col)  := false.B
        wsStageData(row)(col)    := wsPartialSum(row)(col)
        wsStageContext(row)(col) := targetContext
        wsStageMRow(row)(col)    := aStepMRow(row)(col)
      } else {
        val upstreamContext = wsPsumContext(row - 1)(col)
        wsPartialSum(row)(col)   := (wsPsumData(row - 1)(col).asSInt +
          product.pad(wsPsumBits)).asUInt
        wsBypassValid(row)(col)  := wsPsumValid(row - 1)(col) &&
          contextWsMode(upstreamContext) &&
          row.U >= contextTotalK(upstreamContext)
        wsStageData(row)(col)    := Mux(
          wsComputeValid(row)(col),
          wsPartialSum(row)(col),
          wsPsumData(row - 1)(col)
        )
        wsStageContext(row)(col) := Mux(
          wsComputeValid(row)(col),
          targetContext,
          upstreamContext
        )
        wsStageMRow(row)(col)    := Mux(
          wsComputeValid(row)(col),
          aStepMRow(row)(col),
          wsPsumMRow(row - 1)(col)
        )
      }
      wsStageValid(row)(col)   := wsComputeValid(row)(col) || wsBypassValid(row)(
        col
      )
    }
  }

  val wsTerminalValid     = Wire(Vec(tile, Bool()))
  val wsTerminalData      = Wire(Vec(tile, UInt(wsPsumBits.W)))
  val wsTerminalContext   = Wire(Vec(tile, UInt(contextWidth.W)))
  val wsTerminalMRow      = Wire(Vec(tile, UInt(5.W)))
  val wsTerminalKTileKind = Wire(Vec(tile, UInt(2.W)))
  for (col <- 0 until tile) {
    wsTerminalValid(col)     := wsStageValid(tile - 1)(col)
    wsTerminalData(col)      := wsStageData(tile - 1)(col)
    wsTerminalContext(col)   := wsStageContext(tile - 1)(col)
    wsTerminalMRow(col)      := wsStageMRow(tile - 1)(col)
    wsTerminalKTileKind(col) := contextWsKTileKind(wsTerminalContext(col))
  }

  // Column c completes c cycles after column 0. Delay it by 15-c cycles so a
  // complete 512-bit M row reaches the accumulator pipeline in one cycle.
  val wsDeskewData = Seq.tabulate(tile) { col =>
    RegInit(VecInit(Seq.fill(math.max(tile - 1 - col, 1))(0.U(wsPsumBits.W))))
  }

  val wsDeskewValid = Seq.tabulate(tile) { col =>
    RegInit(VecInit(Seq.fill(math.max(tile - 1 - col, 1))(false.B)))
  }

  val wsDeskewContext = Seq.tabulate(tile) { col =>
    RegInit(VecInit(Seq.fill(math.max(tile - 1 - col, 1))(0.U(contextWidth.W))))
  }

  val wsDeskewMRow = Seq.tabulate(tile) { col =>
    RegInit(VecInit(Seq.fill(math.max(tile - 1 - col, 1))(0.U(5.W))))
  }

  val wsDeskewKTileKind = Seq.tabulate(tile) { col =>
    RegInit(
      VecInit(Seq.fill(math.max(tile - 1 - col, 1))(SystolicKTileKind.DIRECT))
    )
  }

  val wsAlignedValid     = Wire(Vec(tile, Bool()))
  val wsAlignedData      = Wire(Vec(tile, UInt(wsPsumBits.W)))
  val wsAlignedContext   = Wire(Vec(tile, UInt(contextWidth.W)))
  val wsAlignedMRow      = Wire(Vec(tile, UInt(5.W)))
  val wsAlignedKTileKind = Wire(Vec(tile, UInt(2.W)))
  for (col <- 0 until tile) {
    val delay = tile - 1 - col
    if (delay == 0) {
      wsAlignedValid(col)     := wsTerminalValid(col)
      wsAlignedData(col)      := wsTerminalData(col)
      wsAlignedContext(col)   := wsTerminalContext(col)
      wsAlignedMRow(col)      := wsTerminalMRow(col)
      wsAlignedKTileKind(col) := wsTerminalKTileKind(col)
    } else {
      wsAlignedValid(col)     := wsDeskewValid(col)(delay - 1)
      wsAlignedData(col)      := wsDeskewData(col)(delay - 1)
      wsAlignedContext(col)   := wsDeskewContext(col)(delay - 1)
      wsAlignedMRow(col)      := wsDeskewMRow(col)(delay - 1)
      wsAlignedKTileKind(col) := wsDeskewKTileKind(col)(delay - 1)
    }
  }

  val wsAlignedRowValid     = wsAlignedValid(0)
  val wsAlignedRowContext   = wsAlignedContext(0)
  val wsAlignedRowIndex     = wsAlignedMRow(0)(rowIndexWidth - 1, 0)
  val wsAlignedRowKTileKind = wsAlignedKTileKind(0)

  val wsAlignedPartialRow = Cat(
    (0 until tile).reverse.map(col => wsAlignedData(col))
  )

  val wsDeskewPending =
    (0 until tile - 1).map(col => wsDeskewValid(col).asUInt.orR).reduce(_ || _)

  val wsReadQueue = Module(
    new Queue(
      new SystolicWsAccumReadEntry(
        contextWidth,
        rowIndexWidth,
        accRowBits,
        wsPsumRowBits
      ),
      2
    )
  )

  val wsResultBuffer = Module(
    new Queue(
      new SystolicWsAccumResultEntry(contextWidth, rowIndexWidth, accRowBits),
      1,
      pipe = true
    )
  )

  val wsReadIssue = Wire(Bool())

  val wsAccReadData = wsAccMem.read(
    wsAccAddress(wsAlignedRowContext, wsAlignedRowIndex),
    wsReadIssue
  )

  val wsReadPartialRow = RegEnable(wsAlignedPartialRow, wsReadIssue)
  val wsReadContext    = RegEnable(wsAlignedRowContext, wsReadIssue)
  val wsReadRow        = RegEnable(wsAlignedRowIndex, wsReadIssue)
  val wsReadKTileKind  = RegEnable(wsAlignedRowKTileKind, wsReadIssue)
  val wsReadPending    = RegNext(wsReadIssue, false.B)

  wsReadQueue.io.enq.valid           := wsReadPending
  wsReadQueue.io.enq.bits.oldRow     := wsAccReadData
  wsReadQueue.io.enq.bits.partialRow := wsReadPartialRow
  wsReadQueue.io.enq.bits.context    := wsReadContext
  wsReadQueue.io.enq.bits.row        := wsReadRow
  wsReadQueue.io.enq.bits.kTileKind  := wsReadKTileKind

  val wsIgnoreOld =
    wsReadQueue.io.deq.bits.kTileKind === SystolicKTileKind.DIRECT ||
      wsReadQueue.io.deq.bits.kTileKind === SystolicKTileKind.FIRST

  val wsAccumulatedElements = Wire(Vec(tile, UInt(accElemBits.W)))
  for (col <- 0 until tile) {
    val oldValue     = wsReadQueue.io.deq.bits
      .oldRow((col + 1) * accElemBits - 1, col * accElemBits)
      .asSInt
    val partialValue = wsReadQueue.io.deq.bits
      .partialRow((col + 1) * wsPsumBits - 1, col * wsPsumBits)
      .asSInt
      .pad(accElemBits)
    wsAccumulatedElements(col) := Mux(
      wsIgnoreOld,
      partialValue,
      oldValue + partialValue
    ).asUInt
  }

  wsResultBuffer.io.enq.valid          := wsReadQueue.io.deq.valid
  wsResultBuffer.io.enq.bits.data      := Cat(wsAccumulatedElements.reverse)
  wsResultBuffer.io.enq.bits.context   := wsReadQueue.io.deq.bits.context
  wsResultBuffer.io.enq.bits.row       := wsReadQueue.io.deq.bits.row
  wsResultBuffer.io.enq.bits.kTileKind := wsReadQueue.io.deq.bits.kTileKind
  wsReadQueue.io.deq.ready             := wsResultBuffer.io.enq.ready

  val wsReadReservations = wsReadQueue.io.count +& wsReadPending.asUInt
  val wsReadCanReserve   = wsReadReservations < 2.U || wsReadQueue.io.deq.fire
  val wsAccumCanAdvance  = !wsAlignedRowValid || wsReadCanReserve

  val activeInputsReady = (0 until contextCount)
    .map(context => !contextActive(context) || contextInputsReady(context))
    .reduce(_ && _)

  pipelineAdvance := (anyContextActive || wsDeskewPending) && activeInputsReady &&
    wsAccumCanAdvance
  wsReadIssue     := pipelineAdvance && wsAlignedRowValid

  when(pipelineAdvance) {
    for (col <- 0 until tile) {
      val delay = tile - 1 - col
      if (delay > 0) {
        wsDeskewData(col)(0)      := wsTerminalData(col)
        wsDeskewValid(col)(0)     := wsTerminalValid(col)
        wsDeskewContext(col)(0)   := wsTerminalContext(col)
        wsDeskewMRow(col)(0)      := wsTerminalMRow(col)
        wsDeskewKTileKind(col)(0) := wsTerminalKTileKind(col)
        for (stage <- 1 until delay) {
          wsDeskewData(col)(stage)      := wsDeskewData(col)(stage - 1)
          wsDeskewValid(col)(stage)     := wsDeskewValid(col)(stage - 1)
          wsDeskewContext(col)(stage)   := wsDeskewContext(col)(stage - 1)
          wsDeskewMRow(col)(stage)      := wsDeskewMRow(col)(stage - 1)
          wsDeskewKTileKind(col)(stage) := wsDeskewKTileKind(col)(stage - 1)
        }
      }
    }

    for (row <- 0 until tile) {
      for (col <- 0 until tile) {
        aPipeData(row)(col)    := aStepData(row)(col)
        aPipeValid(row)(col)   := aStepValid(row)(col)
        aPipeContext(row)(col) := aStepContext(row)(col)
        aPipeMRow(row)(col)    := aStepMRow(row)(col)

        when(anyContextActive && !contextWsMode(activeContext)) {
          bPipeData(row)(col)    := bStepData(row)(col)
          bPipeValid(row)(col)   := bStepValid(row)(col)
          bPipeContext(row)(col) := bStepContext(row)(col)
        }

        wsPsumValid(row)(col) := false.B

        assert(
          !(wsComputeValid(row)(col) && wsBypassValid(row)(col)),
          "SystolicArrayEX: WS compute and bypass collided in one PE"
        )

        when(wsStageValid(row)(col)) {
          val targetContext = aStepContext(row)(col)
          when(wsComputeValid(row)(col)) {
            val targetWeightBank = contextWeightGeneration(targetContext).asUInt

            assert(
              wsWeightBankValid(targetWeightBank),
              "SystolicArrayEX: WS used an invalid weight bank"
            )
            assert(
              wsWeightBankValidN(targetWeightBank) === contextValidN(
                targetContext
              ) &&
                wsWeightBankValidK(targetWeightBank).pad(progressWidth) ===
                contextTotalK(targetContext),
              "SystolicArrayEX: WS weight bank metadata does not match its context"
            )

            if (row == 0) {
              // No upstream partial sum exists in the first physical PE row.
            } else {
              assert(
                wsPsumValid(row - 1)(col),
                "SystolicArrayEX: WS partial sum did not arrive from the previous PE row"
              )
              assert(
                wsPsumContext(row - 1)(col) === targetContext,
                "SystolicArrayEX: WS partial-sum context changed between PE rows"
              )
              assert(
                wsPsumMRow(row - 1)(col) === aStepMRow(row)(col),
                "SystolicArrayEX: WS partial-sum M row changed between PE rows"
              )
            }
          }

          wsPsumData(row)(col)    := wsStageData(row)(col)
          wsPsumValid(row)(col)   := true.B
          wsPsumContext(row)(col) := wsStageContext(row)(col)
          wsPsumMRow(row)(col)    := wsStageMRow(row)(col)
        }.elsewhen(aStepValid(row)(col) && bStepValid(row)(col)) {
          assert(
            aStepContext(row)(col) === bStepContext(row)(col),
            "SystolicArrayEX: A/B context tags do not match"
          )
          val product         = aStepData(row)(col).asSInt * bStepData(row)(col).asSInt
          val targetContext   = aStepContext(row)(col)
          val targetOsAccBank = contextOsAccBank(targetContext)
          assert(
            osAccBankAllocated(targetOsAccBank),
            "SystolicArrayEX: OS context used an unallocated accumulator bank"
          )
          assert(
            osAccBankOwner(targetOsAccBank) === targetContext,
            "SystolicArrayEX: OS accumulator bank owner mismatch"
          )
          osAcc(targetOsAccBank)(row)(col) :=
            osAcc(targetOsAccBank)(row)(col) + product.pad(accElemBits).asUInt
        }
      }
    }

    for (context <- 0 until contextCount) {
      when(contextActive(context)) {
        val firstCompleteCycle = contextValidN(context).pad(progressWidth) +
          contextTotalK(context) - 2.U
        val osLastCycle        = contextValidM(context).pad(progressWidth) +
          contextValidN(context).pad(progressWidth) + contextTotalK(
            context
          ) - 3.U
        val wsLastCycle        = contextValidM(context).pad(progressWidth) +
          contextValidN(context).pad(progressWidth) + tile.U(
            progressWidth.W
          ) - 3.U
        val lastCycle          = Mux(contextWsMode(context), wsLastCycle, osLastCycle)

        when(
          !contextWsMode(context) && contextFinalSeen(context) &&
            contextAge(context) >= firstCompleteCycle &&
            contextRowsComplete(context) < contextValidM(context)
        ) {
          contextRowsComplete(context) := contextRowsComplete(context) + 1.U
        }

        when(contextAge(context) >= lastCycle) {
          contextActive(context) := false.B
          contextAge(context)    := 0.U
        }.otherwise {
          contextAge(context) := contextAge(context) + 1.U
        }
      }
    }

  }

  when(pipelineAdvance) {
    when(wsAlignedRowValid) {
      for (col <- 0 until tile) {
        when(col.U < contextValidN(wsAlignedRowContext)) {
          assert(
            wsAlignedValid(col),
            "SystolicArrayEX: WS de-skew lost a valid result column"
          )
          assert(
            wsAlignedContext(col) === wsAlignedRowContext &&
              wsAlignedMRow(col) === wsAlignedMRow(0) &&
              wsAlignedKTileKind(col) === wsAlignedRowKTileKind,
            "SystolicArrayEX: WS de-skew mixed rows or contexts"
          )
        }.otherwise {
          assert(
            !wsAlignedValid(col),
            "SystolicArrayEX: WS de-skew produced a column outside valid N"
          )
        }
      }
    }
  }
  when(wsReadPending) {
    assert(
      wsReadQueue.io.enq.ready,
      "SystolicArrayEX: WS accumulator read-response queue overflow"
    )
  }
}
