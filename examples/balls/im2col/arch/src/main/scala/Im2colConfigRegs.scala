package examples.balls.im2col

import chisel3._
import chisel3.util._
import framework.balldomain.rs.BallRsIssue
import framework.top.GlobalConfig

class Im2colConfigRegs(
  val b: GlobalConfig,
  maxIter: Int,
  maxKSize: Int,
  maxPadding: Int
) extends Module {

  private val kW   = log2Ceil(maxKSize + 1)
  private val iterW = b.frontend.iter_len
  private val bankEntries = b.memDomain.bankEntries
  private val tile = 16

  val io = IO(new Bundle {
    val cmd      = Input(new BallRsIssue(b))
    val load     = Input(Bool())
    val invalid  = Output(Bool())
    val robId    = Output(UInt(log2Up(b.frontend.rob_entries).W))
    val isSub    = Output(Bool())
    val subRobId = Output(UInt(log2Up(b.frontend.sub_rob_depth * 4).W))
    val rBank    = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val wBank    = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val iter     = Output(UInt(iterW.W))
    val kSize    = Output(UInt(kW.W))
    val stride   = Output(UInt(8.W))
    val padding  = Output(UInt(8.W))
  })

  private val robId    = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSub    = RegInit(false.B)
  private val subRobId = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val rBank    = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val wBank    = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val iter     = RegInit(0.U(iterW.W))
  private val kSize    = RegInit(0.U(kW.W))
  private val stride   = RegInit(1.U(8.W))
  private val padding  = RegInit(0.U(8.W))

  val cmdIter    = io.cmd.cmd.iter
  val cmdKSize   = io.cmd.cmd.special(7, 0)
  val cmdStride  = io.cmd.cmd.special(15, 8)
  val cmdPadding = io.cmd.cmd.special(23, 16)
  val cmdRBank   = io.cmd.cmd.op1_bank
  val cmdWBank   = io.cmd.cmd.wr_bank

  when(io.load) {
    robId    := io.cmd.rob_id
    isSub    := io.cmd.is_sub
    subRobId := io.cmd.sub_rob_id
    rBank    := cmdRBank
    wBank    := cmdWBank
    iter     := cmdIter
    kSize    := cmdKSize(kW - 1, 0)
    stride   := cmdStride
    padding  := cmdPadding
  }

  val paddedSize = cmdIter +& (cmdPadding << 1)
  val shapeOk = (cmdIter >= 1.U) && (cmdIter <= maxIter.U) &&
    (cmdKSize >= 1.U) && (cmdKSize <= maxKSize.U) &&
    (cmdStride >= 1.U) &&
    (cmdPadding <= maxPadding.U) &&
    (paddedSize >= cmdKSize) &&
    (cmdRBank =/= cmdWBank)

  val outDim = Mux(shapeOk, ((paddedSize - cmdKSize) / cmdStride) + 1.U, 0.U)
  val mElems = outDim * outDim
  val kElems = cmdKSize * cmdKSize
  val mTiles = (mElems +& (tile - 1).U) / tile.U
  val kTiles = (kElems +& (tile - 1).U) / tile.U
  val outputRows = mTiles * kTiles * tile.U
  val footprintOk = shapeOk && (outputRows <= bankEntries.U)

  io.invalid := !footprintOk
  io.robId    := Mux(io.load, io.cmd.rob_id, robId)
  io.isSub    := Mux(io.load, io.cmd.is_sub, isSub)
  io.subRobId := Mux(io.load, io.cmd.sub_rob_id, subRobId)
  io.rBank    := Mux(io.load, cmdRBank, rBank)
  io.wBank    := Mux(io.load, cmdWBank, wBank)
  io.iter     := Mux(io.load, cmdIter, iter)
  io.kSize    := Mux(io.load, cmdKSize(kW - 1, 0), kSize)
  io.stride   := Mux(io.load, cmdStride, stride)
  io.padding  := Mux(io.load, cmdPadding, padding)
}
