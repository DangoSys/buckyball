package examples.balls.smatmul

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig

@instantiable
class SMatMulUnit(val b: GlobalConfig) extends Module {
  private val tile          = 16
  private val resultWords   = 4
  private val addressWidth  = log2Ceil(b.memDomain.bankEntries)
  private val bankWidth     = log2Ceil(b.memDomain.bankNum)
  private val maxOutputRows = b.memDomain.bankEntries / resultWords
  private val accDepth      = maxOutputRows.max(32)
  private val accAddrWidth  = log2Ceil(accDepth)

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "SMatMulBall")
    .getOrElse(throw new IllegalArgumentException("SMatMulBall not found in config"))

  private val osFunct = b.ballDomain.ballISA
    .find(_.mnemonic == "SMATMUL_OS")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("SMATMUL_OS not found in ballISA"))

  private val biasFunct = b.ballDomain.ballISA
    .find(_.mnemonic == "SMATMUL_BIAS")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("SMATMUL_BIAS not found in ballISA"))

  require(mapping.inBW == 2, "SMatMulBall requires inBW=2")
  require(mapping.outBW == 1, "SMatMulBall requires outBW=1")
  require(b.memDomain.bankWidth == 128, "SMatMulBall requires 128-bit bank rows")
  require(b.memDomain.bankMaskLen == 16, "SMatMulBall requires sixteen byte enables")
  require(b.memDomain.bankEntries % resultWords == 0, "SMatMulBall bank depth must be divisible by four")
  require(maxOutputRows >= tile, "SMatMulBall accumulator must hold at least one 16-row tile")
  require((osFunct >> 4) == 4, "SMATMUL_OS must encode two reads and one write")
  require((biasFunct >> 4) == 1, "SMATMUL_BIAS must encode one read")

  @public val io = IO(new Bundle {
    val cmdReq       = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp      = Decoupled(new BallRsComplete(b))
    val bankRead     = Vec(mapping.inBW, Flipped(new BankRead(b)))
    val bankWrite    = Vec(mapping.outBW, Flipped(new BankWrite(b)))
    val channelReady = Input(Bool())
    val status       = new BallStatus
  })

  private val Seq(
    idle,
    waitForChannels,
    biasReadReq,
    biasReadResp,
    initAccumulator,
    loadTile,
    runArray,
    readAccumulator,
    writeAccumulator,
    readResult,
    holdResult,
    writeResult,
    waitForCWrite,
    complete
  ) = Enum(14)

  private val state = RegInit(idle)

  private val robId    = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSub    = RegInit(false.B)
  private val subRobId = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))

  private val biasCommand = RegInit(false.B)
  private val firstBlock  = RegInit(false.B)
  private val lastBlock   = RegInit(false.B)
  private val vectorMode  = RegInit(false.B)
  private val biasValid   = RegInit(false.B)
  private val chainLive   = RegInit(false.B)
  private val biasBank    = RegInit(0.U(bankWidth.W))
  private val biasRow     = RegInit(0.U(2.W))
  private val biasWords   = Reg(Vec(resultWords, UInt(128.W)))

  private val aBank      = RegInit(0.U(bankWidth.W))
  private val bBank      = RegInit(0.U(bankWidth.W))
  private val cBank      = RegInit(0.U(bankWidth.W))
  private val chainRows  = RegInit(0.U(12.W))
  private val chainCBank = RegInit(0.U(bankWidth.W))

  private val outputTileCount    = RegInit(0.U(8.W))
  private val reductionTileCount = RegInit(0.U(8.W))
  private val outputTile         = RegInit(0.U(8.W))
  private val reductionTile      = RegInit(0.U(8.W))
  private val accumulatorRow     = RegInit(0.U(4.W))
  private val resultRow          = RegInit(0.U(4.W))
  private val outputWord         = RegInit(0.U(2.W))
  private val aRowsRequested     = RegInit(0.U(5.W))
  private val aRowsStored        = RegInit(0.U(5.W))
  private val bRowsRequested     = RegInit(0.U(5.W))
  private val bRowsStored        = RegInit(0.U(5.W))
  private val cRowData           = Reg(UInt(512.W))

  private val aRows       = Reg(Vec(tile, UInt(128.W)))
  private val bRows       = Reg(Vec(tile, UInt(128.W)))
  private val accumulator = SyncReadMem(accDepth, UInt(512.W))
  private val array: Instance[Array] = Instantiate(new Array)

  private val computeGlobalRow = outputTile * tile.U + accumulatorRow
  private val resultGlobalRow  = outputTile * tile.U + resultRow
  private val accumulatorRead  = state === readAccumulator || state === readResult
  private val accumulatorWrite = state === initAccumulator || state === writeAccumulator

  private val accumulatorAddress = Mux(
    state === readResult,
    resultGlobalRow(accAddrWidth - 1, 0),
    computeGlobalRow(accAddrWidth - 1, 0)
  )

  private val biasData             = Cat(biasWords.reverse)
  private val accumulatorWriteData = Wire(UInt(512.W))

  private val accumulatorData = accumulator.readWrite(
    accumulatorAddress,
    accumulatorWriteData,
    accumulatorRead || accumulatorWrite,
    accumulatorWrite
  )

  private val accumulatedResult = Wire(Vec(tile, UInt(32.W)))
  for (column <- 0 until tile) {
    val oldValue = accumulatorData(32 * column + 31, 32 * column).asSInt
    val newValue = array.io.result(accumulatorRow)(32 * column + 31, 32 * column).asSInt
    accumulatedResult(column) := (oldValue + newValue).asUInt
  }
  accumulatorWriteData := Mux(state === initAccumulator, biasData, Cat(accumulatedResult.reverse))
  assert(!(accumulatorRead && accumulatorWrite), "SMatMulBall accumulator SRAM is single-port")

  private val aTileLine = Mux(
    vectorMode,
    reductionTile,
    (outputTile * reductionTileCount + reductionTile) << 4
  )

  private val bTileLine = reductionTile << 4
  private val cLine     = (resultGlobalRow << 2) + outputWord

  for (port <- 0 until mapping.inBW) {
    io.bankRead(port).rob_id           := robId
    io.bankRead(port).ball_id          := 0.U
    io.bankRead(port).bank_id          := Mux(port.U === 0.U, aBank, bBank)
    io.bankRead(port).group_id         := 0.U
    io.bankRead(port).io.req.valid     := false.B
    io.bankRead(port).io.req.bits.addr := 0.U
    io.bankRead(port).io.resp.ready    := false.B
  }
  io.bankRead(0).bank_id := Mux(state === biasReadReq || state === biasReadResp, biasBank, aBank)
  io.bankRead(0).io.req.valid     := Mux(
    state === biasReadReq,
    true.B,
    state === loadTile && aRowsRequested < Mux(vectorMode, 1.U, tile.U)
  )
  io.bankRead(0).io.req.bits.addr := Mux(
    state === biasReadReq,
    biasRow.pad(addressWidth),
    (aTileLine + aRowsRequested)(addressWidth - 1, 0)
  )
  io.bankRead(0).io.resp.ready    := state === biasReadResp || (state === loadTile && aRowsStored < tile.U)
  io.bankRead(1).io.req.valid     := state === loadTile && bRowsRequested < tile.U
  io.bankRead(1).io.req.bits.addr := (bTileLine + bRowsRequested)(addressWidth - 1, 0)
  io.bankRead(1).io.resp.ready    := state === loadTile && bRowsStored < tile.U

  private val cWords = cRowData.asTypeOf(Vec(resultWords, UInt(128.W)))
  io.bankWrite(0).rob_id           := robId
  io.bankWrite(0).ball_id          := 0.U
  io.bankWrite(0).bank_id          := cBank
  io.bankWrite(0).group_id         := 0.U
  io.bankWrite(0).io.req.valid     := state === writeResult
  io.bankWrite(0).io.req.bits.addr := cLine(addressWidth - 1, 0)
  io.bankWrite(0).io.req.bits.data := cWords(outputWord)
  io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  io.bankWrite(0).io.resp.ready    := state === waitForCWrite

  array.io.start := state === loadTile &&
    aRowsStored === Mux(vectorMode, 1.U, tile.U) && bRowsStored === tile.U
  array.io.aRows := aRows
  array.io.bRows := bRows

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := robId
  io.cmdResp.bits.is_sub     := isSub
  io.cmdResp.bits.sub_rob_id := subRobId
  io.status.idle             := state === idle
  io.status.running          := state =/= idle && state =/= complete

  when(io.cmdReq.fire) {
    val command   = io.cmdReq.bits.cmd
    val isBias    = command.funct7 === biasFunct.U
    val isMatmul  = command.funct7 === osFunct.U
    val rows      = command.rs2(11, 0)
    val columns   = command.rs2(23, 12)
    val reduction = command.iter
    val first     = command.rs2(24)
    val last      = command.rs2(25)

    assert(isBias || isMatmul, "SMatMulBall received an unknown funct7")
    assert(command.rs1(9, 0) < b.memDomain.bankNum.U, "SMatMulBall input bank 0 is invalid")

    robId       := io.cmdReq.bits.rob_id
    isSub       := io.cmdReq.bits.is_sub
    subRobId    := io.cmdReq.bits.sub_rob_id
    biasCommand := isBias

    when(isBias) {
      assert(command.rs1(29, 10) === 0.U, "SMATMUL_BIAS reserves bank1 and bank2")
      assert(command.iter === 4.U, "SMATMUL_BIAS iter must be four rows")
      assert(command.rs2 === 0.U, "SMATMUL_BIAS requires rs2=0")
      assert(
        command.op1_col === 1.U && command.op2_col === 0.U && command.wr_col === 0.U,
        "SMATMUL_BIAS requires exactly one input bank"
      )
      assert(!chainLive, "SMATMUL_BIAS cannot replace bias during an accumulation chain")
      biasBank := command.op1_bank
      biasRow  := 0.U
    }.otherwise {
      assert(command.rs1(19, 10) < b.memDomain.bankNum.U, "SMATMUL_OS input bank 1 is invalid")
      assert(command.rs1(29, 20) < b.memDomain.bankNum.U, "SMATMUL_OS output bank is invalid")
      assert(
        command.op1_bank =/= command.op2_bank && command.op1_bank =/= command.wr_bank &&
          command.op2_bank =/= command.wr_bank,
        "SMATMUL_OS requires distinct A, B, and C banks"
      )
      assert(
        command.op1_col === 1.U && command.op2_col === 1.U && command.wr_col === 1.U,
        "SMATMUL_OS requires one physical bank per operand"
      )
      assert(
        rows === 1.U || (rows =/= 0.U && rows(3, 0) === 0.U),
        "SMATMUL_OS M must be one or a positive multiple of 16"
      )
      assert(columns === tile.U, "SMATMUL_OS N must be 16")
      assert(reduction =/= 0.U && reduction(3, 0) === 0.U, "SMATMUL_OS K must be a positive multiple of 16")
      assert(command.rs2(63, 26) === 0.U, "SMATMUL_OS reserves rs2[63:26]")
      assert(
        Mux(rows === 1.U, reduction >> 4, (rows >> 4) * reduction) <= b.memDomain.bankEntries.U,
        "SMATMUL_OS A footprint exceeds bank depth"
      )
      assert(reduction <= b.memDomain.bankEntries.U, "SMATMUL_OS B footprint exceeds bank depth")
      assert(rows * resultWords.U <= b.memDomain.bankEntries.U, "SMATMUL_OS C footprint exceeds bank depth")
      when(first) {
        assert(!chainLive, "SMATMUL_OS first block cannot start while a chain is live")
        assert(biasValid, "SMATMUL_OS first block requires a bias preload")
        chainLive  := true.B
        chainRows  := rows
        chainCBank := command.wr_bank
      }.otherwise {
        assert(chainLive, "SMATMUL_OS continuation requires a live chain")
        assert(
          rows === chainRows && command.wr_bank === chainCBank,
          "SMATMUL_OS continuation changed M or C destination"
        )
      }

      firstBlock         := first
      lastBlock          := last
      vectorMode         := rows === 1.U
      aBank              := command.op1_bank
      bBank              := command.op2_bank
      cBank              := command.wr_bank
      outputTileCount    := Mux(rows === 1.U, 1.U, rows >> 4)
      reductionTileCount := reduction >> 4
      outputTile         := 0.U
      reductionTile      := 0.U
      accumulatorRow     := 0.U
      resultRow          := 0.U
      outputWord         := 0.U
      aRowsRequested     := 0.U
      aRowsStored        := 0.U
      bRowsRequested     := 0.U
      bRowsStored        := 0.U
      for (row <- 1 until tile) {
        aRows(row) := 0.U
      }
    }
    state := waitForChannels
  }

  when(state === waitForChannels && io.channelReady) {
    state := Mux(biasCommand, biasReadReq, Mux(firstBlock, initAccumulator, loadTile))
  }

  when(state === biasReadReq && io.bankRead(0).io.req.fire) {
    state := biasReadResp
  }
  when(state === biasReadResp && io.bankRead(0).io.resp.fire) {
    biasWords(biasRow) := io.bankRead(0).io.resp.bits.data
    when(biasRow === (resultWords - 1).U) {
      biasValid := true.B
      state     := complete
    }.otherwise {
      biasRow := biasRow + 1.U
      state   := biasReadReq
    }
  }

  when(state === initAccumulator) {
    when(accumulatorRow === Mux(vectorMode, 0.U, (tile - 1).U)) {
      accumulatorRow := 0.U
      when(outputTile + 1.U === outputTileCount) {
        outputTile     := 0.U
        reductionTile  := 0.U
        aRowsRequested := 0.U
        aRowsStored    := 0.U
        bRowsRequested := 0.U
        bRowsStored    := 0.U
        state          := loadTile
      }.otherwise {
        outputTile := outputTile + 1.U
      }
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
    when(accumulatorRow === Mux(vectorMode, 0.U, (tile - 1).U)) {
      accumulatorRow := 0.U
      when(reductionTile + 1.U < reductionTileCount) {
        reductionTile  := reductionTile + 1.U
        aRowsRequested := 0.U
        aRowsStored    := 0.U
        bRowsRequested := 0.U
        bRowsStored    := 0.U
        state          := loadTile
      }.elsewhen(outputTile + 1.U < outputTileCount) {
        outputTile     := outputTile + 1.U
        reductionTile  := 0.U
        aRowsRequested := 0.U
        aRowsStored    := 0.U
        bRowsRequested := 0.U
        bRowsStored    := 0.U
        state          := loadTile
      }.otherwise {
        when(lastBlock) {
          outputTile := 0.U
          resultRow  := 0.U
          state      := readResult
        }.otherwise {
          state := complete
        }
      }
    }.otherwise {
      accumulatorRow := accumulatorRow + 1.U
      state          := readAccumulator
    }
  }

  when(state === readResult) {
    outputWord := 0.U
    state      := holdResult
  }
  when(state === holdResult) {
    cRowData := accumulatorData
    state    := writeResult
  }
  when(state === writeResult && io.bankWrite(0).io.req.fire) {
    state := waitForCWrite
  }
  when(state === waitForCWrite && io.bankWrite(0).io.resp.fire) {
    when(outputWord =/= (resultWords - 1).U) {
      outputWord := outputWord + 1.U
      state      := writeResult
    }.elsewhen(resultRow =/= Mux(vectorMode, 0.U, (tile - 1).U)) {
      resultRow := resultRow + 1.U
      state     := readResult
    }.elsewhen(outputTile + 1.U < outputTileCount) {
      outputTile := outputTile + 1.U
      resultRow  := 0.U
      state      := readResult
    }.otherwise {
      chainLive := false.B
      state     := complete
    }
  }

  when(state === complete && io.cmdResp.fire) {
    state := idle
  }
}
