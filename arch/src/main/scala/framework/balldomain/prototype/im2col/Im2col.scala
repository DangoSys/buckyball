package framework.balldomain.prototype.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.balldomain.prototype.im2col.configs.Im2colBallParam
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig

@instantiable
class Im2col(val b: GlobalConfig) extends Module {
  private val maxK = Im2colBallParam().InputNum

  private val map = b.ballDomain.ballIdMappings
    .find(_.ballName == "Im2colBall")
    .getOrElse(throw new IllegalArgumentException("Im2colBall not found in config"))

  private val inBW  = map.inBW
  private val outBW = map.outBW

  @public val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  require(inBW >= 1, "[Im2col] inBW must be >= 1")
  require(outBW >= 1, "[Im2col] outBW must be >= 1")

  val cfg = Module(new Im2colConfigRegs(b, maxK))
  val win = Module(new Im2colWindow(maxK))
  val lineBuf: Instance[LineBufferManager] = Instantiate(new LineBufferManager(b))
  val writer:  Instance[StreamWriter]      = Instantiate(new StreamWriter(b))

  val running       = RegInit(false.B)
  val linesReady    = RegInit(false.B)
  val finishPending = RegInit(false.B)
  val respPending   = RegInit(false.B)
  val elemDone      = RegInit(false.B)

  cfg.io.cmd  := io.cmdReq.bits
  cfg.io.load := io.cmdReq.fire
  val invalid = cfg.io.invalid

  io.cmdReq.ready            := !running && !respPending
  io.cmdResp.valid           := respPending
  io.cmdResp.bits.rob_id     := cfg.io.robId
  io.cmdResp.bits.is_sub     := cfg.io.isSub
  io.cmdResp.bits.sub_rob_id := cfg.io.subRobId
  io.status.idle             := !running && !respPending
  io.status.running          := running

  when(io.cmdReq.fire) {
    running       := !invalid
    linesReady    := false.B
    finishPending := false.B
    respPending   := invalid
    elemDone      := false.B
  }
  when(io.cmdResp.fire) {
    respPending := false.B
  }

  win.io.init     := io.cmdReq.fire
  win.io.nextCol  := false.B
  win.io.nextRow  := false.B
  win.io.kRow     := cfg.io.kRow
  win.io.kCol     := cfg.io.kCol
  win.io.inRow    := cfg.io.inRow
  win.io.inCol    := cfg.io.inCol
  win.io.startRow := cfg.io.startRow
  win.io.startCol := cfg.io.startCol
  win.io.colStep  := cfg.io.colStep
  val cmdStart    = io.cmdReq.fire && !invalid
  val loadNextRow = WireDefault(false.B)
  val canEmitElem = running && linesReady && !finishPending && !elemDone
  win.io.elemFire := canEmitElem && writer.io.elemIn.ready

  for (i <- 0 until inBW) {
    lineBuf.io.bankRead(i) <> io.bankRead(i)
  }
  lineBuf.io.startPreload    := cmdStart
  lineBuf.io.startLoadNext   := loadNextRow
  lineBuf.io.kRow            := cfg.io.kRow
  lineBuf.io.inCol           := cfg.io.inCol
  lineBuf.io.rowPtr          := win.io.rowPtr
  lineBuf.io.rBaseBeat       := 0.U
  lineBuf.io.rBankId         := cfg.io.rBank
  lineBuf.io.robId           := cfg.io.robId
  lineBuf.io.elemReq.kRowIdx := win.io.kRowIdx
  lineBuf.io.elemReq.kColIdx := win.io.kColIdx
  lineBuf.io.elemReq.colPtr  := win.io.colPtr

  for (i <- 0 until outBW) {
    writer.io.bankWrite(i) <> io.bankWrite(i)
  }
  writer.io.start        := false.B
  writer.io.init         := cmdStart
  writer.io.flush        := false.B
  writer.io.wBaseBeat    := 0.U
  writer.io.wBankId      := cfg.io.wBank
  writer.io.robId        := cfg.io.robId
  writer.io.elemIn.valid := canEmitElem
  writer.io.elemIn.bits  := lineBuf.io.elemData

  when((cmdStart || loadNextRow) && !lineBuf.io.loadDone) {
    linesReady := false.B
  }.elsewhen(running && !linesReady && !finishPending && lineBuf.io.loadDone) {
    linesReady := true.B
  }

  when(win.io.elemLast && win.io.elemFire) {
    elemDone := true.B
  }

  val windowDone = running && linesReady && elemDone && !writer.io.busy
  when(windowDone && win.io.lastWindow) {
    writer.io.flush := true.B
    linesReady      := false.B
    finishPending   := true.B
    elemDone        := false.B
  }.elsewhen(windowDone && win.io.colEnd) {
    win.io.nextRow := true.B
    loadNextRow    := true.B
    linesReady     := false.B
    elemDone       := false.B
  }.elsewhen(windowDone) {
    win.io.nextCol := true.B
    elemDone       := false.B
  }

  when(finishPending && !writer.io.busy) {
    finishPending := false.B
    running       := false.B
    respPending   := true.B
  }
}
