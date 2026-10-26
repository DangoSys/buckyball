package examples.balls.smatmul

import framework.acc.balldomain.configs.BallRouteParam

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import framework.acc.balldomain.ball.blink.{BallCommand, BallResponse}
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.balldomain.ball.blink.BallStatus
import framework.acc.memdomain.{BankRead, BankWrite}
import examples.balls.smatmul.configs.SMatMulBallParam

@instantiable
class SMatMulUnit(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val routingParam:  BallDomainParam,
  val instance:      BallRouteParam,
  val ballParam:     SMatMulBallParam)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits
  val param           = ballParam
  val rows            = param.tileRows
  val cols            = param.tileCols
  val resultWords     = 4
  val addressWidth    = log2Ceil(memParam.bankEntries)
  val bankIdWidth     = frontendParam.bank_id_len
  val maxOutputRows   = memParam.bankEntries / resultWords
  val accDepth        = maxOutputRows.max(32)
  val accAddrWidth    = log2Ceil(accDepth)

  val mapping = instance

  val osFunct = instance.instructions
    .find(_.mnemonic == "SMATMUL_OS").get.funct7

  val biasFunct = instance.instructions
    .find(_.mnemonic == "SMATMUL_BIAS").get.funct7

  require(mapping.inBW == 2, "SMatMulBall requires inBW=2")
  require(mapping.outBW == 1, "SMatMulBall requires outBW=1")
  require(memParam.bankWidth == 128, "SMatMulBall requires 128-bit bank rows")
  require(
    memParam.bankMaskLen == 16,
    "SMatMulBall requires sixteen byte enables"
  )
  require(
    memParam.bankEntries % resultWords == 0,
    "SMatMulBall bank depth must be divisible by four"
  )
  require(maxOutputRows >= rows, "SMatMulBall accumulator must hold at least one core")
  require(isPow2(memParam.bankEntries), "SMatMulBall requires power-of-two bank depth")
  require((osFunct >> 4) == 4, "SMATMUL_OS must encode two reads and one write")
  require((biasFunct >> 4) == 1, "SMATMUL_BIAS must encode one read")

  @public
  val io = IO(new Bundle {

    val cmdReq = Flipped(Decoupled(new BallCommand(frontendParam, memParam)))

    val cmdResp = Decoupled(new BallResponse(frontendParam))

    val bankRead = Vec(
      mapping.inBW,
      Flipped(new BankRead(
        frontendParam,
        memParam
      ))
    )

    val bankWrite = Vec(
      mapping.outBW,
      Flipped(new BankWrite(
        frontendParam,
        memParam
      ))
    )

    val status = new BallStatus
  })

  val Seq(
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

  val state = RegInit(idle)

  val robId = RegInit(0.U(log2Up(frontendParam.rob_entries).W))

  val biasCommand = RegInit(false.B)
  val firstBlock  = RegInit(false.B)
  val lastBlock   = RegInit(false.B)
  val vectorMode  = RegInit(false.B)
  val biasValid   = RegInit(false.B)
  val chainLive   = RegInit(false.B)
  val biasBank    = RegInit(0.U(bankIdWidth.W))
  val biasBase    = RegInit(0.U(log2Ceil(memParam.bankEntries).W))
  val biasRow     = RegInit(0.U(2.W))
  val biasWords   = Reg(Vec(resultWords, UInt(128.W)))

  val aBank      = RegInit(0.U(bankIdWidth.W))
  val bBank      = RegInit(0.U(bankIdWidth.W))
  val cBank      = RegInit(0.U(bankIdWidth.W))
  val bases      = Reg(Vec(3, UInt(memParam.groupIdWidth.W)))
  // A and B group g hold reduction block g; C rows continue across groups.
  val kGroups    = RegInit(0.U(log2Up(memParam.bankNum + 1).W))
  val kGroup     = RegInit(0.U(log2Up(memParam.bankNum + 1).W))
  val cBase      = RegInit(0.U(addressWidth.W))
  val chainRows  = RegInit(0.U(12.W))
  val chainCBank = RegInit(0.U(bankIdWidth.W))
  val chainCBase = RegInit(0.U(addressWidth.W))

  val outputTileCount    = RegInit(0.U(8.W))
  val reductionTileCount = RegInit(0.U(8.W))
  val outputTile         = RegInit(0.U(8.W))
  val reductionTile      = RegInit(0.U(8.W))
  val accumulatorRow     = RegInit(0.U(4.W))
  val resultRow          = RegInit(0.U(4.W))
  val outputWord         = RegInit(0.U(2.W))
  val aRowsRequested     = RegInit(0.U(5.W))
  val aRowsStored        = RegInit(0.U(5.W))
  val bRowsRequested     = RegInit(0.U(5.W))
  val bRowsStored        = RegInit(0.U(5.W))
  val cRowData           = Reg(UInt(512.W))

  val aRows       = Reg(Vec(rows, UInt(128.W)))
  val bRows       = Reg(Vec(cols, UInt(128.W)))
  val accumulator = SyncReadMem(accDepth, UInt(512.W))
  val array: Instance[Array] = Instantiate(new Array(ballParam))

  val computeGlobalRow = outputTile * rows.U + accumulatorRow
  val resultGlobalRow  = outputTile * rows.U + resultRow
  val accumulatorRead  = state === readAccumulator || state === readResult
  val accumulatorWrite = state === initAccumulator || state === writeAccumulator

  val accumulatorAddress = Mux(
    state === readResult,
    resultGlobalRow(accAddrWidth - 1, 0),
    computeGlobalRow(accAddrWidth - 1, 0)
  )

  val biasData             = Cat(biasWords.reverse)
  val accumulatorWriteData = Wire(UInt(512.W))

  val accumulatorData = accumulator.readWrite(
    accumulatorAddress,
    accumulatorWriteData,
    accumulatorRead || accumulatorWrite,
    accumulatorWrite
  )

  val accumulatedResult = Wire(Vec(cols, UInt(32.W)))
  for (column <- 0 until cols) {
    val oldValue = accumulatorData(32 * column + 31, 32 * column).asSInt
    val newValue = array.io.result(accumulatorRow)(32 * column + 31, 32 * column).asSInt
    accumulatedResult(column) := (oldValue + newValue).asUInt
  }
  accumulatorWriteData := Mux(state === initAccumulator, biasData, Cat(accumulatedResult.reverse))
  assert(!(accumulatorRead && accumulatorWrite), "SMatMulBall accumulator SRAM is single-port")

  val aTileLine = Mux(
    vectorMode,
    reductionTile,
    (outputTile * reductionTileCount + reductionTile) * rows.U
  )

  val bTileLine = reductionTile * cols.U
  val cLine     = cBase +& (resultGlobalRow << 2) +& outputWord

  for (port <- 0 until mapping.inBW) {
    io.bankRead(port).rob_id           := robId
    io.bankRead(port).bank_id          := Mux(port.U === 0.U, aBank, bBank)
    io.bankRead(port).group_id         := bases(port) + kGroup
    io.bankRead(port).io.req.valid     := false.B
    io.bankRead(port).io.req.bits.addr := 0.U
    io.bankRead(port).io.resp.ready    := false.B
  }
  io.bankRead(0).bank_id := Mux(state === biasReadReq || state === biasReadResp, biasBank, aBank)
  io.bankRead(0).io.req.valid     := Mux(
    state === biasReadReq,
    true.B,
    state === loadTile && aRowsRequested < Mux(vectorMode, 1.U, rows.U)
  )
  io.bankRead(0).io.req.bits.addr := Mux(
    state === biasReadReq,
    (biasBase +& biasRow)(addressWidth - 1, 0),
    (aTileLine + aRowsRequested)(addressWidth - 1, 0)
  )
  io.bankRead(0).io.resp.ready    := state === biasReadResp || (state === loadTile && aRowsStored < rows.U)
  io.bankRead(1).io.req.valid     := state === loadTile && bRowsRequested < cols.U
  io.bankRead(1).io.req.bits.addr := (bTileLine + bRowsRequested)(addressWidth - 1, 0)
  io.bankRead(1).io.resp.ready    := state === loadTile && bRowsStored < cols.U

  val cWords = cRowData.asTypeOf(Vec(resultWords, UInt(128.W)))
  io.bankWrite(0).rob_id           := robId
  io.bankWrite(0).bank_id          := cBank
  io.bankWrite(0).group_id         := bases(2) + (cLine >> addressWidth)
  io.bankWrite(0).io.req.valid     := state === writeResult
  io.bankWrite(0).io.req.bits.addr := cLine(addressWidth - 1, 0)
  io.bankWrite(0).io.req.bits.data := cWords(outputWord)
  io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  io.bankWrite(0).io.resp.ready    := state === waitForCWrite

  array.io.start := state === loadTile &&
    aRowsStored === Mux(vectorMode, 1.U, rows.U) && bRowsStored === cols.U
  array.io.aRows := aRows
  array.io.bRows := bRows

  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := state === complete
  io.cmdResp.bits.rob_id := robId
  io.status.idle         := state === idle
  io.status.running      := state =/= idle && state =/= complete

  when(io.cmdReq.fire) {
    val command = io.cmdReq.bits
    bases := VecInit(command.op1_group, command.op2_group, command.wr_group)
    val isBias    = command.funct === biasFunct.U
    val isMatmul  = command.funct === osFunct.U
    val rows      = command.rs2(11, 0)
    val columns   = command.rs2(23, 12)
    val reduction = command.iter
    val first     = command.rs2(24)
    val last      = command.rs2(25)

    assert(isBias || isMatmul, "SMatMulBall received an unknown funct7")

    robId       := io.cmdReq.bits.rob_id
    biasCommand := isBias
    kGroups     := command.op1_col
    kGroup      := 0.U

    when(isBias) {
      assert(command.iter === 4.U, "SMATMUL_BIAS iter must be four rows")
      assert(command.rs2(63, 6) === 0.U, "SMATMUL_BIAS reserves rs2[63:6]")
      assert(
        command.rs2(5, 0) +& 4.U <= memParam.bankEntries.U,
        "SMATMUL_BIAS inputBase plus four rows must fit one bank"
      )
      assert(
        command.op1_col === 1.U && command.op2_col === 0.U && command.wr_col === 0.U,
        "SMATMUL_BIAS requires exactly one input bank"
      )
      assert(!chainLive, "SMATMUL_BIAS cannot replace bias during an accumulation chain")
      biasBank := command.op1_bank
      biasBase := command.rs2(5, 0)
      biasRow  := 0.U
    }.otherwise {
      assert(
        command.op1_bank =/= command.op2_bank && command.op1_bank =/= command.wr_bank &&
          command.op2_bank =/= command.wr_bank,
        "SMATMUL_OS requires distinct A, B, and C banks"
      )
      assert(
        command.op1_col =/= 0.U && command.op1_col === command.op2_col &&
          command.op1_col <= memParam.bankNum.U && command.wr_col =/= 0.U,
        "SMATMUL_OS A and B must have the same reduction groups"
      )
      assert(rows <= accDepth.U, "SMATMUL_OS M exceeds the accumulator")
      assert(
        rows === 1.U || (rows =/= 0.U && rows(2, 0) === 0.U),
        "SMATMUL_OS M must be one or a positive multiple of 8"
      )
      assert(columns === cols.U, "SMATMUL_OS N must be 16")
      assert(reduction =/= 0.U && reduction(3, 0) === 0.U, "SMATMUL_OS K must be a positive multiple of 16")
      assert(command.rs2(63, 32) === 0.U, "SMATMUL_OS reserves rs2[63:32]")
      assert(
        Mux(
          rows === 1.U,
          reduction >> 4,
          (rows * reduction) >> 4
        ) <= memParam.bankEntries.U,
        "SMATMUL_OS A footprint exceeds bank depth"
      )
      assert(
        reduction <= memParam.bankEntries.U,
        "SMATMUL_OS B footprint exceeds bank depth"
      )
      assert(
        command.rs2(
          31,
          26
        ) +& rows * resultWords.U <= command.wr_col * memParam.bankEntries.U,
        "SMATMUL_OS C footprint exceeds its groups"
      )
      when(first) {
        assert(!chainLive, "SMATMUL_OS first block cannot start while a chain is live")
        assert(biasValid, "SMATMUL_OS first block requires a bias preload")
        chainLive  := true.B
        chainRows  := rows
        // A chain retains its architectural destination across bank-version renaming.
        chainCBank := command.wr_arch_bank
        chainCBase := command.rs2(31, 26)
      }.otherwise {
        assert(chainLive, "SMATMUL_OS continuation requires a live chain")
        assert(
          rows === chainRows && command.wr_arch_bank === chainCBank &&
            command.rs2(31, 26) === chainCBase,
          "SMATMUL_OS continuation changed M or C destination/base"
        )
      }

      firstBlock         := first
      lastBlock          := last
      vectorMode         := rows === 1.U
      aBank              := command.op1_bank
      bBank              := command.op2_bank
      cBank              := command.wr_bank
      cBase              := command.rs2(31, 26)
      outputTileCount    := Mux(rows === 1.U, 1.U, rows / param.tileRows.U)
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
      for (row <- 1 until 8) {
        aRows(row) := 0.U
      }
    }
    state := waitForChannels
  }

  when(state === waitForChannels) {
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
    when(accumulatorRow === Mux(vectorMode, 0.U, (rows - 1).U)) {
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
    when(aRowsStored === Mux(vectorMode, 1.U, rows.U) && bRowsStored === cols.U) {
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
    when(accumulatorRow === Mux(vectorMode, 0.U, (rows - 1).U)) {
      accumulatorRow := 0.U
      when(reductionTile + 1.U < reductionTileCount) {
        reductionTile  := reductionTile + 1.U
        aRowsRequested := 0.U
        aRowsStored    := 0.U
        bRowsRequested := 0.U
        bRowsStored    := 0.U
        state          := loadTile
      }.elsewhen(kGroup + 1.U < kGroups) {
        kGroup         := kGroup + 1.U
        reductionTile  := 0.U
        aRowsRequested := 0.U
        aRowsStored    := 0.U
        bRowsRequested := 0.U
        bRowsStored    := 0.U
        state          := loadTile
      }.elsewhen(outputTile + 1.U < outputTileCount) {
        outputTile     := outputTile + 1.U
        kGroup         := 0.U
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
    }.elsewhen(resultRow =/= Mux(vectorMode, 0.U, (rows - 1).U)) {
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
