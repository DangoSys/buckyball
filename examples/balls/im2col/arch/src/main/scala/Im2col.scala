package examples.balls.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import examples.balls.im2col.configs.Im2colBallParam
import examples.balls.common.RestoringDiv
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig

@instantiable
class Im2col(val b: GlobalConfig) extends Module {
  private val ballCfg  = Im2colBallParam(b)
  private val maxIter  = ballCfg.maxIter
  private val maxKSize = ballCfg.maxKSize
  private val maxPad   = ballCfg.maxPadding
  private val tile     = 16

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "Im2colBall")
    .getOrElse(throw new IllegalArgumentException("Im2colBall not found in config"))

  private val im2colFunct = b.ballDomain.ballISA
    .find(_.mnemonic == "IM2COL")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("IM2COL not found in ballISA"))

  require(mapping.inBW == 1, "Im2colBall requires inBW=1")
  require(mapping.outBW == 1, "Im2colBall requires outBW=1")
  require(b.memDomain.bankWidth == 128, "Im2colBall requires 128-bit bank rows")
  require(isPow2(b.memDomain.bankEntries), "Im2colBall requires power-of-two bank depth")
  require(maxIter + 2 * maxPad <= 255, "Im2colBall dimensions must fit the divider")
  require((im2colFunct >> 4) == 3, "IM2COL must encode one read and one write")

  @public val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(mapping.inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(mapping.outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  private val cfg = Module(new Im2colConfigRegs(b, maxIter, maxKSize, maxPad))
  private val win = Module(new Im2colWindow(maxKSize))
  private val div = Module(new RestoringDiv(16, 8))
  private val lineBuf: Instance[LineBufferManager] = Instantiate(new LineBufferManager(b))
  private val writer:  Instance[StreamWriter]      = Instantiate(new StreamWriter(b))

  private val cIdle :: cDivRow :: cDivCol :: cCheck :: cDivStart :: cLaunch :: Nil = Enum(6)
  private val calc                                                                 = RegInit(cIdle)

  private val running     = RegInit(false.B)
  private val inputReady  = RegInit(false.B)
  private val respPending = RegInit(false.B)
  private val armed       = RegInit(false.B)
  private val outRowsReg  = RegInit(0.U(16.W))
  private val outColsReg  = RegInit(0.U(16.W))
  private val startOutRow = RegInit(0.U(16.W))
  private val startOutCol = RegInit(0.U(16.W))
  private val localWindow = RegInit(0.U(12.W))
  private val elemAge     = RegInit(0.U(2.W))
  private val launch      = WireDefault(false.B)

  cfg.io.cmd  := io.cmdReq.bits
  cfg.io.load := io.cmdReq.fire

  io.cmdReq.ready            := !running && !respPending && !armed && calc === cIdle
  io.cmdResp.valid           := respPending
  io.cmdResp.bits.rob_id     := cfg.io.robId
  io.cmdResp.bits.is_sub     := cfg.io.isSub
  io.cmdResp.bits.sub_rob_id := cfg.io.subRobId
  io.status.idle             := !running && !respPending && !armed && calc === cIdle
  io.status.running          := running || armed || calc =/= cIdle

  div.io.start := false.B
  div.io.a     := 0.U
  div.io.b     := 0.U

  when(io.cmdReq.fire) {
    val command = io.cmdReq.bits.cmd
    assert(command.funct7 === im2colFunct.U, "Im2colBall received an unknown funct7")
    assert(command.rs1(9, 0) < b.memDomain.bankNum.U, "IM2COL input bank is invalid")
    assert(command.rs1(19, 10) === 0.U, "IM2COL reserves input bank 1")
    assert(command.rs1(29, 20) < b.memDomain.bankNum.U, "IM2COL output bank is invalid")
    assert(command.rs2(63) === 0.U, "IM2COL reserves rs2[63]")
    assert(command.iter(b.frontend.iter_len - 1, 16) === 0.U, "IM2COL input size exceeds 16 bits")
    assert(
      command.op1_col === 1.U && command.op2_col === 0.U && command.wr_col === 1.U,
      "IM2COL requires one physical input bank and one physical output bank"
    )
    assert(command.rs2(7, 0) > 0.U && command.rs2(7, 0) <= maxKSize.U, "IM2COL kernel must be in the configured range")
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
    val inputRows   = (inputValues +& (tile - 1).U) >> log2Ceil(tile)
    val windows     = outRowsReg * outColsReg
    val windowEnd   = cfg.io.windowStart +& cfg.io.windowCount
    val kernelElems = cfg.io.kernel * cfg.io.kernel
    val mTiles      = (cfg.io.windowCount +& (tile - 1).U) >> log2Ceil(tile)
    val kTiles      = (kernelElems +& (tile - 1).U) >> log2Ceil(tile)
    val outputRows  = mTiles * kTiles * tile.U

    assert(cfg.io.windowStart < windows && windowEnd <= windows, "IM2COL selected window range exceeds the output shape")
    assert(inputRows <= b.memDomain.bankEntries.U, "IM2COL input footprint exceeds bank depth")
    assert(outputRows <= b.memDomain.bankEntries.U, "IM2COL output footprint exceeds bank depth")

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
  private val canEmitElem = running && inputReady && elemAge === 3.U
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
  lineBuf.io.robId     := cfg.io.robId

  writer.io.bankWrite(0) <> io.bankWrite(0)
  writer.io.init         := launch
  writer.io.wBankId      := cfg.io.wBank
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
