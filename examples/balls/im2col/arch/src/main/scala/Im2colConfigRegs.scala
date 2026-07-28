package examples.balls.im2col

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
    val iter    = Output(UInt(b.frontend.iter_len.W))
    val kSize   = Output(UInt(log2Ceil(maxK + 1).W))
    val stride  = Output(UInt(8.W))
    val padding = Output(UInt(8.W))
  })

  private val robId    = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSub    = RegInit(false.B)
  private val subRobId = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val rBank    = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val wBank    = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val iter    = RegInit(0.U(b.frontend.iter_len.W))
  private val kSize   = RegInit(0.U(log2Ceil(maxK + 1).W))
  private val stride  = RegInit(1.U(8.W))
  private val padding = RegInit(0.U(8.W))

  val cmdIter    = io.cmd.cmd.iter
  val cmdKSize   = io.cmd.cmd.special(7, 0)
  val cmdStride  = io.cmd.cmd.special(15, 8)
  val cmdPadding = io.cmd.cmd.special(23, 16)

  when(io.load) {
    robId    := io.cmd.rob_id
    isSub    := io.cmd.is_sub
    subRobId := io.cmd.sub_rob_id
    rBank    := io.cmd.cmd.op1_bank
    wBank    := io.cmd.cmd.wr_bank
    iter    := cmdIter
    kSize   := cmdKSize
    stride  := cmdStride
    padding := cmdPadding
  }

  val paddedSize = cmdIter +& (cmdPadding << 1)
  io.invalid  := (cmdIter === 0.U) || (cmdIter > maxK.U) ||
    (cmdKSize === 0.U) || (cmdKSize > maxK.U) ||
    (cmdStride === 0.U) || (paddedSize < cmdKSize)
  io.robId    := Mux(io.load, io.cmd.rob_id, robId)
  io.isSub    := Mux(io.load, io.cmd.is_sub, isSub)
  io.subRobId := Mux(io.load, io.cmd.sub_rob_id, subRobId)
  io.rBank    := Mux(io.load, io.cmd.cmd.op1_bank, rBank)
  io.wBank    := Mux(io.load, io.cmd.cmd.wr_bank, wBank)
  io.iter    := Mux(io.load, cmdIter, iter)
  io.kSize   := Mux(io.load, cmdKSize, kSize)
  io.stride  := Mux(io.load, cmdStride, stride)
  io.padding := Mux(io.load, cmdPadding, padding)
}
