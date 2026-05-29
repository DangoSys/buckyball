package framework.balldomain.prototype.im2col

import chisel3._
import chisel3.util._

class LineLoadCtrl(maxK: Int, maxWords: Int) extends Module {

  val io = IO(new Bundle {
    val startPreload  = Input(Bool())
    val startLoadNext = Input(Bool())
    val kRow          = Input(UInt(log2Ceil(maxK + 1).W))
    val inColWords    = Input(UInt(log2Ceil(maxWords + 1).W))
    val inCol         = Input(UInt(16.W))
    val rowPtr        = Input(UInt(16.W))
    val rBaseBeat     = Input(UInt(32.W))
    val targetSlot    = Input(UInt(log2Ceil(maxK).W))
    val reqReady      = Input(Bool())
    val respValid     = Input(Bool())
    val reqValid      = Output(Bool())
    val reqAddr       = Output(UInt(32.W))
    val respReady     = Output(Bool())
    val writeRow      = Output(UInt(log2Ceil(maxK).W))
    val writeBeat     = Output(UInt(log2Ceil(maxWords).W))
    val advanceFifo   = Output(Bool())
    val done          = Output(Bool())
  })

  private val active    = RegInit(false.B)
  private val loadNext  = RegInit(false.B)
  private val row       = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val beat      = RegInit(0.U(log2Ceil(maxWords + 1).W))
  private val pending   = RegInit(false.B)
  private val writeRow  = RegInit(0.U(log2Ceil(maxK).W))
  private val writeBeat = RegInit(0.U(log2Ceil(maxWords).W))

  when(io.startPreload) {
    active   := true.B
    loadNext := false.B
    row      := 0.U
    beat     := 0.U
    pending  := false.B
  }.elsewhen(io.startLoadNext) {
    active   := true.B
    loadNext := true.B
    row      := 0.U
    beat     := 0.U
    pending  := false.B
  }

  val rowsDone = Mux(loadNext, row === 1.U, row === io.kRow)
  val canIssue = active && !rowsDone && !pending && beat < io.inColWords
  val srcRow   = Mux(loadNext, io.rowPtr + io.kRow - 1.U, io.rowPtr + row)
  val srcByte  = srcRow * io.inCol + beat * 16.U
  io.reqValid  := canIssue
  io.reqAddr   := io.rBaseBeat + (srcByte >> 4)
  io.respReady := pending
  io.writeRow  := writeRow
  io.writeBeat := writeBeat
  io.done      := !active

  when(canIssue && io.reqReady) {
    pending   := true.B
    writeRow  := Mux(loadNext, io.targetSlot, row)(log2Ceil(maxK) - 1, 0)
    writeBeat := beat(log2Ceil(maxWords) - 1, 0)
  }

  val respFire = pending && io.respValid
  val lastBeat = beat + 1.U === io.inColWords
  io.advanceFifo := respFire && loadNext && lastBeat

  when(respFire) {
    pending := false.B
    when(lastBeat) {
      beat := 0.U
      row  := row + 1.U
      when(Mux(loadNext, row === 0.U, row + 1.U === io.kRow)) {
        active := false.B
      }
    }.otherwise {
      beat := beat + 1.U
    }
  }
}
