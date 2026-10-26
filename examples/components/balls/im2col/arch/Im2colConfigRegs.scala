package examples.balls.im2col

import chisel3.experimental.hierarchy.{instantiable, public}

import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import examples.balls.im2col.configs.Im2colBallParam

import chisel3._
import framework.acc.balldomain.ball.blink.BallCommand
import chisel3.util._

@instantiable
class Im2colConfigRegs(
  frontendParam:  FrontendParam,
  memParam:       MemDomainParam,
  val maxIter:    Int,
  val maxKSize:   Int,
  val maxPadding: Int,
  val ballParam:  Im2colBallParam)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits

  val kW = log2Ceil(maxKSize + 1)

  @public
  val io = IO(new Bundle {

    val cmd = Input(new BallCommand(frontendParam, memParam))

    val load        = Input(Bool())
    val invalid     = Output(Bool())
    val robId       = Output(UInt(log2Up(frontendParam.rob_entries).W))
    val rBank       = Output(UInt(frontendParam.bank_id_len.W))
    val wBank       = Output(UInt(frontendParam.bank_id_len.W))
    val rGroup      = Output(UInt(memParam.groupIdWidth.W))
    val wGroup      = Output(UInt(memParam.groupIdWidth.W))
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

  val robId       = RegInit(0.U(log2Up(frontendParam.rob_entries).W))
  val rBank       = RegInit(0.U(frontendParam.bank_id_len.W))
  val wBank       = RegInit(0.U(frontendParam.bank_id_len.W))
  val rGroup      = RegInit(0.U(memParam.groupIdWidth.W))
  val wGroup      = RegInit(0.U(memParam.groupIdWidth.W))
  val inputSize   = RegInit(0.U(16.W))
  val kernel      = RegInit(0.U(kW.W))
  val stride      = RegInit(0.U(8.W))
  val padding     = RegInit(0.U(8.W))
  val startRow    = RegInit(0.U(8.W))
  val startCol    = RegInit(0.U(8.W))
  val inputBase   = RegInit(0.U(6.W))
  val lane        = RegInit(0.U(4.W))
  val windowStart = RegInit(0.U(6.W))
  val windowCount = RegInit(0.U(7.W))

  when(io.load) {
    robId       := io.cmd.rob_id
    rBank       := io.cmd.op1_bank
    wBank       := io.cmd.wr_bank
    rGroup      := io.cmd.op1_group
    wGroup      := io.cmd.wr_group
    inputSize   := io.cmd.iter
    kernel      := io.cmd.rs2(7, 0)(kW - 1, 0)
    stride      := io.cmd.rs2(15, 8)
    padding     := io.cmd.rs2(23, 16)
    startCol    := io.cmd.rs2(31, 24)
    startRow    := io.cmd.rs2(39, 32)
    inputBase   := io.cmd.rs2(45, 40)
    lane        := io.cmd.rs2(49, 46)
    windowStart := io.cmd.rs2(55, 50)
    windowCount := io.cmd.rs2(62, 56)
  }

  val padded = inputSize +& (padding << 1)
  io.invalid := inputSize === 0.U || inputSize > maxIter.U ||
    kernel === 0.U || kernel > maxKSize.U || stride === 0.U ||
    padding > maxPadding.U || startRow > padding || startCol > padding ||
    inputBase +& inputSize * inputSize > memParam.bankEntries.U ||
    padded < kernel + startRow || padded < kernel + startCol ||
    windowCount === 0.U || rBank === wBank

  io.robId       := robId
  io.rBank       := rBank
  io.wBank       := wBank
  io.rGroup      := rGroup
  io.wGroup      := wGroup
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
