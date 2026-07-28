package framework.balldomain.prototype.systolicarray

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig

abstract class SystolicArrayEXDatapath(b: GlobalConfig) extends SystolicArrayEXInput(b) {
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
}
