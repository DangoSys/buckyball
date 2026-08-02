package examples.balls.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import examples.balls.im2col.configs.Im2colBallParam
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig

@instantiable
class Im2col(val b: GlobalConfig) extends Module {
  private val ballCfg  = Im2colBallParam(b)
  private val maxIter  = ballCfg.maxIter
  private val maxKSize = ballCfg.maxKSize
  private val maxPad   = ballCfg.maxPadding

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

  val cfg = Module(new Im2colConfigRegs(b, maxIter, maxKSize, maxPad))
  val win = Module(new Im2colWindow(maxKSize))
  val lineBuf: Instance[LineBufferManager] = Instantiate(new LineBufferManager(b))
  val writer:  Instance[StreamWriter]      = Instantiate(new StreamWriter(b))

  val running     = RegInit(false.B)
  val inputReady  = RegInit(false.B)
  val respPending = RegInit(false.B)

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
    running     := !invalid
    inputReady  := false.B
    respPending := invalid
  }
  when(io.cmdResp.fire) {
    respPending := false.B
  }

  win.io.init     := io.cmdReq.fire
  win.io.next     := false.B
  win.io.iter     := cfg.io.iter
  win.io.kSize    := cfg.io.kSize
  win.io.stride   := cfg.io.stride
  win.io.padding  := cfg.io.padding
  val cmdStart    = io.cmdReq.fire && !invalid
  val canEmitElem = running && inputReady
  win.io.elemFire := canEmitElem && writer.io.elemIn.ready

  for (i <- 0 until inBW) {
    lineBuf.io.bankRead(i) <> io.bankRead(i)
  }
  lineBuf.io.start   := cmdStart
  lineBuf.io.iter    := cfg.io.iter
  lineBuf.io.stride  := cfg.io.stride
  lineBuf.io.padding := cfg.io.padding
  lineBuf.io.outRow  := win.io.outRow
  lineBuf.io.outCol  := win.io.outCol
  lineBuf.io.kRowIdx := win.io.kRowIdx
  lineBuf.io.kColIdx := win.io.kColIdx
  lineBuf.io.rBankId := cfg.io.rBank
  lineBuf.io.robId   := cfg.io.robId

  for (i <- 0 until outBW) {
    writer.io.bankWrite(i) <> io.bankWrite(i)
  }
  writer.io.init         := cmdStart
  writer.io.wBankId      := cfg.io.wBank
  writer.io.robId        := cfg.io.robId
  writer.io.elemIn.valid := canEmitElem
  writer.io.elemIn.bits  := lineBuf.io.elemData
  writer.io.elemLast     := win.io.elemLast
  writer.io.lastWindow   := win.io.last
  writer.io.kSize        := cfg.io.kSize
  val outputDim = ((cfg.io.iter +& (cfg.io.padding << 1) - cfg.io.kSize) /
    cfg.io.stride) + 1.U
  writer.io.windowIdx := win.io.outRow * outputDim + win.io.outCol

  when(cmdStart) {
    inputReady := false.B
  }.elsewhen(running && !inputReady && lineBuf.io.loadDone) {
    inputReady := true.B
  }

  when(writer.io.opComplete) {
    inputReady  := false.B
    running     := false.B
    respPending := true.B
  }.elsewhen(writer.io.windowComplete) {
    win.io.next := true.B
  }
}
