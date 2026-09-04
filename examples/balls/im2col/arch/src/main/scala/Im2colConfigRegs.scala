package examples.balls.im2col

import chisel3._
import chisel3.util._
import framework.balldomain.rs.BallRsIssue
import framework.top.GlobalConfig

class Im2colConfigRegs(
  val b:      GlobalConfig,
  maxIter:    Int,
  maxKSize:   Int,
  maxPadding: Int)
    extends Module {

  private val kW = log2Ceil(maxKSize + 1)

  val io = IO(new Bundle {
    val cmd         = Input(new BallRsIssue(b))
    val load        = Input(Bool())
    val invalid     = Output(Bool())
    val robId       = Output(UInt(log2Up(b.frontend.rob_entries).W))
    val isSub       = Output(Bool())
    val subRobId    = Output(UInt(log2Up(b.frontend.sub_rob_depth * 4).W))
    val rBank       = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val wBank       = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val inputSize   = Output(UInt(16.W))
    val kernel      = Output(UInt(kW.W))
    val stride      = Output(UInt(8.W))
    val padding     = Output(UInt(8.W))
    val startRow    = Output(UInt(8.W))
    val startCol    = Output(UInt(8.W))
    val inputBase   = Output(UInt(6.W))
    val lane        = Output(UInt(4.W))
    val windowStart = Output(UInt(6.W))
    val windowCount = Output(UInt(7.W))
  })

  private val robId       = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSub       = RegInit(false.B)
  private val subRobId    = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val rBank       = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val wBank       = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val inputSize   = RegInit(0.U(16.W))
  private val kernel      = RegInit(0.U(kW.W))
  private val stride      = RegInit(0.U(8.W))
  private val padding     = RegInit(0.U(8.W))
  private val startRow    = RegInit(0.U(8.W))
  private val startCol    = RegInit(0.U(8.W))
  private val inputBase   = RegInit(0.U(6.W))
  private val lane        = RegInit(0.U(4.W))
  private val windowStart = RegInit(0.U(6.W))
  private val windowCount = RegInit(0.U(7.W))

  when(io.load) {
    robId       := io.cmd.rob_id
    isSub       := io.cmd.is_sub
    subRobId    := io.cmd.sub_rob_id
    rBank       := io.cmd.cmd.op1_bank
    wBank       := io.cmd.cmd.wr_bank
    inputSize   := io.cmd.cmd.iter
    kernel      := io.cmd.cmd.rs2(7, 0)(kW - 1, 0)
    stride      := io.cmd.cmd.rs2(15, 8)
    padding     := io.cmd.cmd.rs2(23, 16)
    startCol    := io.cmd.cmd.rs2(31, 24)
    startRow    := io.cmd.cmd.rs2(39, 32)
    inputBase   := io.cmd.cmd.rs2(45, 40)
    lane        := io.cmd.cmd.rs2(49, 46)
    windowStart := io.cmd.cmd.rs2(55, 50)
    windowCount := io.cmd.cmd.rs2(62, 56)
  }

  private val padded = inputSize +& (padding << 1)
  io.invalid := inputSize === 0.U || inputSize > maxIter.U ||
    kernel === 0.U || kernel > maxKSize.U || stride === 0.U ||
    padding > maxPadding.U || startRow > padding || startCol > padding ||
    inputBase +& inputSize * inputSize > b.memDomain.bankEntries.U ||
    padded < kernel + startRow || padded < kernel + startCol ||
    windowCount === 0.U || rBank === wBank

  io.robId       := robId
  io.isSub       := isSub
  io.subRobId    := subRobId
  io.rBank       := rBank
  io.wBank       := wBank
  io.inputSize   := inputSize
  io.kernel      := kernel
  io.stride      := stride
  io.padding     := padding
  io.startRow    := startRow
  io.startCol    := startCol
  io.inputBase   := inputBase
  io.lane        := lane
  io.windowStart := windowStart
  io.windowCount := windowCount
}
