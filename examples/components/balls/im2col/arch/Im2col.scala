package examples.balls.im2col

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
import examples.balls.im2col.configs.Im2colBallParam

@instantiable
class Im2col(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val routingParam:  BallDomainParam,
  val instance:      BallRouteParam,
  val ballParam:     Im2colBallParam)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits
  val ballCfg         = ballParam
  val maxIter         = ballCfg.maxIter
  val maxKSize        = ballCfg.maxKSize
  val maxPad          = ballCfg.maxPadding
  val core            = 16

  val mapping = instance

  require(mapping.inBW == 1, "Im2colBall requires inBW=1")
  require(mapping.outBW == 1, "Im2colBall requires outBW=1")
  require(memParam.bankWidth == 128, "Im2colBall requires 128-bit bank rows")
  require(
    isPow2(memParam.bankEntries),
    "Im2colBall requires power-of-two bank depth"
  )
  require(maxIter + 2 * maxPad <= 255, "Im2colBall dimensions must fit the divider")

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

  val cfg =
    Instantiate(new Im2colConfigRegs(frontendParam, memParam, maxIter, maxKSize, maxPad, ballParam))

  val win =
    Instantiate(new Im2colWindow(
      maxKSize,
      ballParam
    ))

  val div = Instantiate(new RestoringDiv(16, 8, ballParam))

  val lineBuf: Instance[LineBufferManager] =
    Instantiate(new LineBufferManager(frontendParam, memParam, routingParam, instance, ballParam))
  val writer:  Instance[StreamWriter]      =
    Instantiate(new StreamWriter(frontendParam, memParam, routingParam, instance, ballParam))

  val cIdle :: cDivRow :: cDivCol :: cCheck :: cDivStart :: cLaunch :: Nil = Enum(6)
  val calc                                                                 = RegInit(cIdle)

  val running     = RegInit(false.B)
  val inputReady  = RegInit(false.B)
  val respPending = RegInit(false.B)
  val armed       = RegInit(false.B)
  val outRowsReg  = RegInit(0.U(16.W))
  val outColsReg  = RegInit(0.U(16.W))
  val startOutRow = RegInit(0.U(16.W))
  val startOutCol = RegInit(0.U(16.W))
  val localWindow = RegInit(0.U(12.W))
  val elemAge     = RegInit(0.U(2.W))
  val launch      = WireDefault(false.B)

  cfg.io.cmd  := io.cmdReq.bits
  cfg.io.load := io.cmdReq.fire

  io.cmdReq.ready        := !running && !respPending && !armed && calc === cIdle
  io.cmdResp.valid       := respPending
  io.cmdResp.bits.rob_id := cfg.io.robId
  io.status.idle         := !running && !respPending && !armed && calc === cIdle
  io.status.running      := running || armed || calc =/= cIdle

  div.io.start := false.B
  div.io.a     := 0.U
  div.io.b     := 0.U

  when(io.cmdReq.fire) {
    val command = io.cmdReq.bits

    assert(command.rs2(63) === 0.U, "IM2COL reserves rs2[63]")
    assert(
      command.iter(frontendParam.iter_len - 1, 16) === 0.U,
      "IM2COL input size exceeds 16 bits"
    )
    assert(
      command.op1_col === 1.U && command.op2_col === 0.U && command.wr_col === 1.U,
      "IM2COL requires one physical input bank and one physical output bank"
    )
    assert(
      command.rs2(7, 0) > 0.U && command.rs2(7, 0) <= maxKSize.U,
      "IM2COL kernel must be in the configured range"
    )
    armed := true.B
  }

  when(armed) {
    armed := false.B
    assert(!cfg.io.invalid, "IM2COL command has an illegal shape or bank assignment")
    val padded = cfg.io.inputSize +& (cfg.io.padding << 1)
    div.io.start := true.B
    div.io.a     := (padded - cfg.io.kernel - cfg.io.startRow)(15, 0)
    div.io.b     := cfg.io.stride
    calc         := cDivRow
  }

  when(calc === cDivRow && div.io.done) {
    assert(div.io.r === 0.U, "IM2COL output row shape must be integral")
    outRowsReg := div.io.q + 1.U
    val padded = cfg.io.inputSize +& (cfg.io.padding << 1)
    div.io.start := true.B
    div.io.a     := (padded - cfg.io.kernel - cfg.io.startCol)(15, 0)
    div.io.b     := cfg.io.stride
    calc         := cDivCol
  }

  when(calc === cDivCol && div.io.done) {
    assert(div.io.r === 0.U, "IM2COL output column shape must be integral")
    outColsReg := div.io.q + 1.U
    calc       := cCheck
  }

  when(calc === cCheck) {
    val inputValues = cfg.io.inputSize * cfg.io.inputSize
    val inputRows   = (inputValues +& (core - 1).U) >> log2Ceil(core)
    val windows     = outRowsReg * outColsReg
    val windowEnd   = cfg.io.windowStart +& cfg.io.windowCount
    val kernelElems = cfg.io.kernel * cfg.io.kernel
    val mTiles      = (cfg.io.windowCount +& (core - 1).U) >> log2Ceil(core)
    val kTiles      = (kernelElems +& (core - 1).U) >> log2Ceil(core)
    val outputRows  = mTiles * kTiles * core.U

    assert(cfg.io.windowStart < windows && windowEnd <= windows, "IM2COL selected window range exceeds the output shape")
    assert(
      inputRows <= memParam.bankEntries.U,
      "IM2COL input footprint exceeds bank depth"
    )
    assert(
      outputRows <= memParam.bankEntries.U,
      "IM2COL output footprint exceeds bank depth"
    )

    div.io.start := true.B
    div.io.a     := cfg.io.windowStart
    div.io.b     := outColsReg(7, 0)
    calc         := cDivStart
  }

  when(calc === cDivStart && div.io.done) {
    assert(div.io.q < outRowsReg && div.io.r < outColsReg, "IM2COL window start decoding exceeded the output shape")
    startOutRow := div.io.q
    startOutCol := div.io.r
    calc        := cLaunch
  }

  when(calc === cLaunch) {
    launch      := true.B
    running     := true.B
    inputReady  := false.B
    localWindow := 0.U
    elemAge     := 0.U
    calc        := cIdle
  }

  when(io.cmdResp.fire) {
    respPending := false.B
  }

  win.io.init        := launch
  win.io.next        := false.B
  win.io.kernel      := cfg.io.kernel
  win.io.outputCols  := outColsReg
  win.io.startRow    := startOutRow
  win.io.startCol    := startOutCol
  win.io.windowCount := cfg.io.windowCount
  val canEmitElem = running && inputReady && elemAge === 3.U
  win.io.elemFire := canEmitElem && writer.io.elemIn.ready

  lineBuf.io.bankRead(0) <> io.bankRead(0)
  lineBuf.io.start     := launch
  lineBuf.io.inRows    := cfg.io.inputSize
  lineBuf.io.inCols    := cfg.io.inputSize
  lineBuf.io.rowStride := cfg.io.stride
  lineBuf.io.colStride := cfg.io.stride
  lineBuf.io.padding   := cfg.io.padding
  lineBuf.io.startRow  := cfg.io.startRow
  lineBuf.io.startCol  := cfg.io.startCol
  lineBuf.io.inputBase := cfg.io.inputBase
  lineBuf.io.lane      := cfg.io.lane
  lineBuf.io.outRow    := win.io.outRow
  lineBuf.io.outCol    := win.io.outCol
  lineBuf.io.kRowIdx   := win.io.kRowIdx
  lineBuf.io.kColIdx   := win.io.kColIdx
  lineBuf.io.rBankId   := cfg.io.rBank
  lineBuf.io.rGroup    := cfg.io.rGroup
  lineBuf.io.robId     := cfg.io.robId

  writer.io.bankWrite(0) <> io.bankWrite(0)
  writer.io.init         := launch
  writer.io.wBankId      := cfg.io.wBank
  writer.io.wGroup       := cfg.io.wGroup
  writer.io.robId        := cfg.io.robId
  writer.io.elemIn.valid := canEmitElem
  writer.io.elemIn.bits  := lineBuf.io.elemData
  writer.io.elemLast     := win.io.elemLast
  writer.io.finalWindow  := win.io.last
  writer.io.kRows        := cfg.io.kernel
  writer.io.kCols        := cfg.io.kernel
  writer.io.windowIdx    := localWindow

  when(launch) {
    inputReady := false.B
    elemAge    := 0.U
  }.elsewhen(running && !inputReady && lineBuf.io.loadDone) {
    inputReady := true.B
    elemAge    := 0.U
  }.elsewhen(writer.io.elemIn.fire || writer.io.windowComplete) {
    elemAge := 0.U
  }.elsewhen(running && inputReady && elemAge =/= 3.U) {
    elemAge := elemAge + 1.U
  }

  when(writer.io.opComplete) {
    inputReady  := false.B
    running     := false.B
    respPending := true.B
  }.elsewhen(writer.io.windowComplete) {
    win.io.next := true.B
    localWindow := localWindow + 1.U
  }

}
