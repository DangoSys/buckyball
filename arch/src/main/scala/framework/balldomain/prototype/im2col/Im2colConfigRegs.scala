package framework.balldomain.prototype.im2col

import chisel3._
import chisel3.util._
import framework.balldomain.rs.BallRsIssue
import framework.top.GlobalConfig

class Im2colConfigRegs(val b: GlobalConfig, maxK: Int) extends Module {

  val io = IO(new Bundle {
    val cmd      = Input(new BallRsIssue(b))
    val load     = Input(Bool())
    val invalid  = Output(Bool())
    val robId    = Output(UInt(log2Up(b.frontend.rob_entries).W))
    val isSub    = Output(Bool())
    val subRobId = Output(UInt(log2Up(b.frontend.sub_rob_depth * 4).W))
    val rBank    = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val wBank    = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val kRow     = Output(UInt(log2Ceil(maxK + 1).W))
    val kCol     = Output(UInt(log2Ceil(maxK + 1).W))
    val inRow    = Output(UInt(16.W))
    val inCol    = Output(UInt(16.W))
    val startRow = Output(UInt(16.W))
    val startCol = Output(UInt(16.W))
  })

  private val robId    = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSub    = RegInit(false.B)
  private val subRobId = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val rBank    = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val wBank    = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val kRow     = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val kCol     = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val inRow    = RegInit(0.U(16.W))
  private val inCol    = RegInit(0.U(16.W))
  private val startRow = RegInit(0.U(16.W))
  private val startCol = RegInit(0.U(16.W))

  val cmdKCol  = io.cmd.cmd.special(3, 0)
  val cmdKRow  = io.cmd.cmd.special(7, 4)
  val cmdInCol = io.cmd.cmd.special(12, 8)
  val cmdInRow = io.cmd.cmd.special(22, 13)

  when(io.load) {
    robId    := io.cmd.rob_id
    isSub    := io.cmd.is_sub
    subRobId := io.cmd.sub_rob_id
    rBank    := io.cmd.cmd.op1_bank
    wBank    := io.cmd.cmd.wr_bank
    kCol     := cmdKCol
    kRow     := cmdKRow
    inCol    := cmdInCol
    inRow    := cmdInRow
    startCol := io.cmd.cmd.special(27, 23)
    startRow := io.cmd.cmd.special(37, 28)
  }

  io.invalid  := (cmdKCol === 0.U) || (cmdKRow === 0.U) ||
    (cmdInCol === 0.U) || (cmdInRow === 0.U) ||
    (cmdInCol < cmdKCol) || (cmdInRow < cmdKRow)
  io.robId    := Mux(io.load, io.cmd.rob_id, robId)
  io.isSub    := Mux(io.load, io.cmd.is_sub, isSub)
  io.subRobId := Mux(io.load, io.cmd.sub_rob_id, subRobId)
  io.rBank    := Mux(io.load, io.cmd.cmd.op1_bank, rBank)
  io.wBank    := Mux(io.load, io.cmd.cmd.wr_bank, wBank)
  io.kRow     := Mux(io.load, cmdKRow, kRow)
  io.kCol     := Mux(io.load, cmdKCol, kCol)
  io.inRow    := Mux(io.load, cmdInRow, inRow)
  io.inCol    := Mux(io.load, cmdInCol, inCol)
  io.startRow := Mux(io.load, io.cmd.cmd.special(37, 28), startRow)
  io.startCol := Mux(io.load, io.cmd.cmd.special(27, 23), startCol)
}
