package examples.balls.smatmul

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig

@instantiable
class SMatMulUnit(val b: GlobalConfig) extends Module {
  private val tile         = 16
  private val addressWidth = log2Up(b.memDomain.bankEntries)
  private val bankWidth    = log2Up(b.memDomain.bankNum)

  private val ballMapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "SMatMulBall")
    .getOrElse(throw new IllegalArgumentException("SMatMulBall not found in config"))

  private val inBW         = ballMapping.inBW
  private val outBW        = ballMapping.outBW
  private val rowWords     = 4
  private val outputRounds = rowWords / outBW

  require(inBW >= 2, "SMatMulBall requires SRAM read ports for A and B")
  require(
    outBW > 0 && outBW <= rowWords && rowWords % outBW == 0,
    "SMatMulBall outBW must divide four 128-bit result words"
  )
  require(b.memDomain.bankWidth == 128, "SMatMulBall requires 128-bit SRAM rows")
  require(b.memDomain.bankMaskLen == 16, "SMatMulBall requires sixteen byte enables")
  require(3 * addressWidth <= b.frontend.iter_len, "SMatMulBall iter cannot hold three base lines")

  @public
  val io = IO(new Bundle {
    val cmdReq       = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp      = Decoupled(new BallRsComplete(b))
    val bankRead     = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite    = Vec(outBW, Flipped(new BankWrite(b)))
    val channelReady = Input(Bool())
    val status       = new BallStatus
  })

  val Seq(
    idle,
    waitForChannels,
    clearAccumulator,
    loadTile,
    runArray,
    readAccumulator,
    writeAccumulator,
    readResult,
    holdResult,
    writeResult,
    waitForCWrite,
    complete
  ) = Enum(12)

  val state = RegInit(idle)

  val robId              = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val isSub              = RegInit(false.B)
  val subRobId           = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  val aBank              = RegInit(0.U(bankWidth.W))
  val bBank              = RegInit(0.U(bankWidth.W))
  val cBank              = RegInit(0.U(bankWidth.W))
  val aBaseLine          = RegInit(0.U(addressWidth.W))
  val bBaseLine          = RegInit(0.U(addressWidth.W))
  val cBaseLine          = RegInit(0.U(addressWidth.W))
  val outputTileCount    = RegInit(0.U(12.W))
  val panelCount         = RegInit(1.U(7.W))
  val panelIndex         = RegInit(0.U(7.W))
  val isWs               = RegInit(false.B)
  val reductionTileCount = RegInit(0.U(12.W))
  val outputTile         = RegInit(0.U(12.W))
  val reductionTile      = RegInit(0.U(12.W))
  val accumulatorRow     = RegInit(0.U(4.W))
  val resultRow          = RegInit(0.U(4.W))
  val outputRound        = RegInit(0.U(math.max(1, log2Up(outputRounds)).W))
  val aRowsRequested     = RegInit(0.U(5.W))
  val aRowsStored        = RegInit(0.U(5.W))
  val bRowsRequested     = RegInit(0.U(5.W))
  val bRowsStored        = RegInit(0.U(5.W))
  val cRowData           = Reg(UInt(512.W))

  val aRows       = Reg(Vec(tile, UInt(128.W)))
  val bRows       = Reg(Vec(tile, UInt(128.W)))
  val accumulator = SyncReadMem(tile, UInt(512.W))
  val array: Instance[Array] = Instantiate(new Array)

  val accumulatorRead      = state === readAccumulator || state === readResult
  val accumulatorWrite     = state === clearAccumulator || state === writeAccumulator
  val accumulatorAddress   = Mux(state === readResult, resultRow, accumulatorRow)
  val accumulatorWriteData = Wire(UInt(512.W))

  val accumulatorData = accumulator.readWrite(
    accumulatorAddress,
    accumulatorWriteData,
    accumulatorRead || accumulatorWrite,
    accumulatorWrite
  )

  assert(!(accumulatorRead && accumulatorWrite), "SMatMulBall accumulator SRAM is single-port")

  val accumulatedResult = Wire(Vec(tile, UInt(32.W)))
  for (column <- 0 until tile) {
    val oldValue = accumulatorData(32 * column + 31, 32 * column).asSInt
    val newValue = array.io.result(accumulatorRow)(32 * column + 31, 32 * column).asSInt
    accumulatedResult(column) := (oldValue + newValue).asUInt
  }
  val accumulatedRow = Cat(accumulatedResult.reverse)
  accumulatorWriteData := Mux(state === clearAccumulator, 0.U(512.W), accumulatedRow)

  val aTileLine = aBaseLine.pad(32) + ((outputTile * reductionTileCount + reductionTile) << 4)
  val bTileLine = bBaseLine.pad(32) + Mux(isWs, panelIndex << 4, reductionTile << 4)
  val wsLine    = panelIndex * (tile * outputRounds).U + resultRow * outputRounds.U + outputRound
  val osLine    = (outputTile * tile.U + resultRow) * outputRounds.U + outputRound
  val cLine     = cBaseLine.pad(32) + Mux(isWs, wsLine, osLine)
  val aReadLine = aTileLine + aRowsRequested
  val bReadLine = bTileLine + bRowsRequested

  for (port <- 0 until inBW) {
    io.bankRead(port).rob_id           := robId
    io.bankRead(port).ball_id          := 0.U
    io.bankRead(port).bank_id          := Mux(port.U === 0.U, aBank, bBank)
    io.bankRead(port).group_id         := 0.U
    io.bankRead(port).io.req.valid     := false.B
    io.bankRead(port).io.req.bits.addr := 0.U
    io.bankRead(port).io.resp.ready    := false.B
  }
  io.bankRead(0).group_id := 0.U
  io.bankRead(0).io.req.valid     := state === loadTile && (!isWs || panelIndex === 0.U) && aRowsRequested < tile.U
  io.bankRead(0).io.req.bits.addr := aReadLine(addressWidth - 1, 0)
  io.bankRead(0).io.resp.ready    := state === loadTile && (!isWs || panelIndex === 0.U) && aRowsStored < tile.U
  io.bankRead(1).group_id         := 0.U
  io.bankRead(1).io.req.valid     := state === loadTile && bRowsRequested < tile.U
  io.bankRead(1).io.req.bits.addr := bReadLine(addressWidth - 1, 0)
  io.bankRead(1).io.resp.ready    := state === loadTile && bRowsStored < tile.U

  val cWords = cRowData.asTypeOf(Vec(rowWords, UInt(128.W)))
  for (port <- 0 until outBW) {
    io.bankWrite(port).rob_id           := robId
    io.bankWrite(port).ball_id          := 0.U
    io.bankWrite(port).bank_id          := cBank
    io.bankWrite(port).group_id         := port.U
    io.bankWrite(port).io.req.bits.addr := cLine(addressWidth - 1, 0)
    io.bankWrite(port).io.req.bits.data := cWords(outputRound * outBW.U + port.U)
    io.bankWrite(port).io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  }
  for (port <- 0 until outBW) {
    io.bankWrite(port).io.req.valid := state === writeResult
  }
  val allCWriteResponses = io.bankWrite.map(_.io.resp.valid).reduce(_ && _)
  for (port <- 0 until outBW) {
    io.bankWrite(port).io.resp.ready := state === waitForCWrite && allCWriteResponses
  }

  array.io.start := state === loadTile && aRowsStored === tile.U && bRowsStored === tile.U
  array.io.aRows := aRows
  array.io.bRows := bRows

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := robId
  io.cmdResp.bits.is_sub     := isSub
  io.cmdResp.bits.sub_rob_id := subRobId

  when(state === clearAccumulator) {
    when(accumulatorRow === 15.U) {
      accumulatorRow := 0.U
      reductionTile  := 0.U
      aRowsRequested := Mux(isWs && panelIndex =/= 0.U, tile.U, 0.U)
      aRowsStored    := Mux(isWs && panelIndex =/= 0.U, tile.U, 0.U)
      bRowsRequested := 0.U
      bRowsStored    := 0.U
      state          := loadTile
    }.otherwise {
      accumulatorRow := accumulatorRow + 1.U
    }
  }

  when(state === loadTile) {
    when(io.bankRead(0).io.req.fire)(aRowsRequested := aRowsRequested + 1.U)
    when(io.bankRead(1).io.req.fire)(bRowsRequested := bRowsRequested + 1.U)
    when(io.bankRead(0).io.resp.fire) {
      aRows(aRowsStored(3, 0)) := io.bankRead(0).io.resp.bits.data
      aRowsStored              := aRowsStored + 1.U
    }
    when(io.bankRead(1).io.resp.fire) {
      bRows(bRowsStored(3, 0)) := io.bankRead(1).io.resp.bits.data
      bRowsStored              := bRowsStored + 1.U
    }
    when(aRowsStored === tile.U && bRowsStored === tile.U) {
      state := runArray
    }
  }

  when(state === runArray && array.io.done) {
    accumulatorRow := 0.U
    state          := readAccumulator
  }

  when(state === readAccumulator) {
    state := writeAccumulator
  }

  when(state === writeAccumulator) {
    when(accumulatorRow === 15.U) {
      when(reductionTile + 1.U === reductionTileCount) {
        resultRow := 0.U
        state     := readResult
      }.otherwise {
        reductionTile  := reductionTile + 1.U
        aRowsRequested := 0.U
        aRowsStored    := 0.U
        bRowsRequested := 0.U
        bRowsStored    := 0.U
        state          := loadTile
      }
    }.otherwise {
      accumulatorRow := accumulatorRow + 1.U
      state          := readAccumulator
    }
  }

  when(state === readResult) {
    outputRound := 0.U
    state       := holdResult
  }

  when(state === holdResult) {
    cRowData := accumulatorData
    state    := writeResult
  }

  when(state === writeResult) {
    val allCWriteRequests = io.bankWrite.map(_.io.req.ready).reduce(_ && _)
    assert(
      io.bankWrite.map(_.io.req.ready).map(_ === io.bankWrite(0).io.req.ready).reduce(_ && _),
      "SMatMulBall C channels must be ready together"
    )
    when(allCWriteRequests) {
      state := waitForCWrite
    }
  }

  when(state === waitForCWrite) {
    assert(
      io.bankWrite.map(_.io.resp.valid).map(_ === io.bankWrite(0).io.resp.valid).reduce(_ && _),
      "SMatMulBall C channels must respond together"
    )
    when(allCWriteResponses) {
      when(outputRound =/= (outputRounds - 1).U) {
        outputRound := outputRound + 1.U
        state       := writeResult
      }.otherwise {
        when(resultRow === 15.U) {
          when(isWs && panelIndex + 1.U < panelCount) {
            panelIndex     := panelIndex + 1.U
            accumulatorRow := 0.U
            resultRow      := 0.U
            bRowsRequested := 0.U
            bRowsStored    := 0.U
            state          := clearAccumulator
          }.elsewhen(outputTile + 1.U === outputTileCount) {
            state := complete
          }.otherwise {
            outputTile     := outputTile + 1.U
            accumulatorRow := 0.U
            state          := clearAccumulator
          }
        }.otherwise {
          resultRow := resultRow + 1.U
          state     := readResult
        }
      }
    }
  }

  when(state === complete && io.cmdResp.fire) {
    state := idle
  }

  when(io.cmdReq.fire) {
    val command     = io.cmdReq.bits.cmd
    val rows        = command.rs2(11, 0)
    val columns     = command.rs2(23, 12)
    val reduction   = command.rs2(35, 24)
    val commandIter = command.iter
    robId              := io.cmdReq.bits.rob_id
    isSub              := io.cmdReq.bits.is_sub
    subRobId           := io.cmdReq.bits.sub_rob_id
    aBank              := command.op1_bank
    bBank              := command.op2_bank
    cBank              := command.wr_bank
    aBaseLine          := commandIter(addressWidth - 1, 0)
    bBaseLine          := commandIter(2 * addressWidth - 1, addressWidth)
    cBaseLine          := commandIter(3 * addressWidth - 1, 2 * addressWidth)
    outputTileCount    := rows >> 4
    panelCount         := columns >> 4
    panelIndex         := 0.U
    reductionTileCount := reduction >> 4
    outputTile         := 0.U
    accumulatorRow     := 0.U
    resultRow          := 0.U
    isWs               := command.funct7 === 68.U
    assert(rows =/= 0.U && rows(3, 0) === 0.U, "SMatMulBall rows must be a non-zero multiple of 16")
    assert(command.funct7 === 65.U || command.funct7 === 68.U, "SMatMulBall funct7 must select OS or WS")
    when(command.funct7 === 68.U) {
      assert(
        rows === 16.U && reduction === 16.U && columns(3, 0) === 0.U &&
          columns * outputRounds.U <= b.memDomain.bankEntries.U,
        "SMATMUL_WS requires rows=k=16 and C to fit in its output groups"
      )
      assert(
        command.op1_col === 1.U && command.op2_col === 1.U && command.wr_col === outBW.U,
        "SMATMUL_WS bank groups mismatch"
      )
    }.otherwise {
      assert(columns === 16.U, "SMATMUL_OS columns must be exactly 16")
      assert(
        command.op1_col === 1.U && command.op2_col === 1.U && command.wr_col === outBW.U,
        "SMATMUL_OS bank groups mismatch"
      )
    }
    assert(reduction =/= 0.U && reduction(3, 0) === 0.U, "SMatMulBall reduction must be a non-zero multiple of 16")
    assert(command.rs2(63, 36) === 0.U, "SMatMulBall rs2[63:36] must be zero")
    assert(commandIter(b.frontend.iter_len - 1, 3 * addressWidth) === 0.U, "SMatMulBall iter high bits must be zero")
    assert(
      command.op1_bank =/= command.op2_bank && command.op1_bank =/= command.wr_bank && command.op2_bank =/= command.wr_bank,
      "SMatMulBall A, B, and C must use different SRAM banks"
    )
    assert(
      commandIter(addressWidth - 1, 0) + (rows >> 4) * (reduction >> 4) * tile.U <= b.memDomain.bankEntries.U,
      "SMatMulBall A does not fit in one bank"
    )
    assert(
      commandIter(2 * addressWidth - 1, addressWidth) +
        Mux(command.funct7 === 68.U, columns, reduction) <= b.memDomain.bankEntries.U,
      "SMatMulBall B does not fit in one bank"
    )
    assert(
      commandIter(3 * addressWidth - 1, 2 * addressWidth) +
        Mux(command.funct7 === 68.U, columns * outputRounds.U, rows * outputRounds.U) <= b.memDomain.bankEntries.U,
      "SMatMulBall C does not fit in its physical banks"
    )
    state              := waitForChannels
  }

  when(state === waitForChannels && io.channelReady) {
    state := clearAccumulator
  }

  io.status.idle    := state === idle
  io.status.running := state =/= idle
}
