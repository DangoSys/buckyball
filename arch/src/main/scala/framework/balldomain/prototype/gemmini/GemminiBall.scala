package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.rs.BallRsComplete
import framework.top.GlobalConfig

@instantiable
class GemminiBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {

  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "GemminiBall")
    .getOrElse(throw new IllegalArgumentException("GemminiBall not found in config"))
  val inBW             = ballCommonConfig.inBW
  val outBW            = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink:  BlinkIO    = io
  def status: BallStatus = io.status

  // =========================================================================
  // Sub-modules
  // =========================================================================
  val exCtrl:         Instance[GemminiExCtrl]      = Instantiate(new GemminiExCtrl(b))
  val matmulUnroller: Instance[LoopMatmulUnroller] = Instantiate(new LoopMatmulUnroller(b))
  val convUnroller:   Instance[LoopConvUnroller]   = Instantiate(new LoopConvUnroller(b))
  val encoder:        Instance[LoopCmdEncoder]     = Instantiate(new LoopCmdEncoder(b))

  // =========================================================================
  // Config registers for Loop modes
  // =========================================================================
  val loopWsConfig   = Reg(new LoopWsConfig(b))
  val loopConvConfig = Reg(new LoopConvWsConfig(b))

  // rob_id tracking for bank metadata
  val rob_id_reg = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  when(io.cmdReq.fire) {
    rob_id_reg := io.cmdReq.bits.rob_id
  }

  // =========================================================================
  // Instruction routing by funct7
  // =========================================================================
  val funct7 = io.cmdReq.bits.cmd.funct7

  val rs2Data = io.cmdReq.bits.cmd.special

  val isConfig     = funct7 === 0x02.U // GEMMINI_CONFIG (enable=000, opcode=2)
  val isPreload    = funct7 === 0x35.U // GEMMINI_PRELOAD (enable=011, opcode=5)
  val isComputePre = funct7 === 0x42.U // GEMMINI_COMPUTE_PRELOADED (enable=100, opcode=2)
  val isComputeAcc = funct7 === 0x43.U // GEMMINI_COMPUTE_ACCUMULATED (enable=100, opcode=3)
  val isFlush      = funct7 === 0x03.U // GEMMINI_FLUSH (enable=000, opcode=3)
  val isExUnit     = isConfig || isPreload || isComputePre || isComputeAcc || isFlush

  val isLoopWsConfig    = funct7 >= 0x50.U && funct7 <= 0x56.U
  val isLoopWsTrigger   = funct7 === 0x57.U
  val isLoopConvConfig  = funct7 >= 0x60.U && funct7 <= 0x68.U
  val isLoopConvTrigger = funct7 === 0x69.U

  // =========================================================================
  // ExUnit path (non-Loop: CONFIG/PRELOAD/COMPUTE/FLUSH)
  // =========================================================================
  exCtrl.exio.cmdReq.valid := io.cmdReq.valid && isExUnit
  exCtrl.exio.cmdReq.bits  := io.cmdReq.bits

  // =========================================================================
  // Config latch path (immediate cmdResp)
  // =========================================================================
  val configRespValid = RegInit(false.B)
  val configRespBits  = Reg(new BallRsComplete(b))
  configRespValid := false.B // default: pulse

  when(io.cmdReq.fire && isLoopWsConfig) {
    configRespValid           := true.B
    configRespBits.rob_id     := io.cmdReq.bits.rob_id
    configRespBits.is_sub     := io.cmdReq.bits.is_sub
    configRespBits.sub_rob_id := io.cmdReq.bits.sub_rob_id
    switch(funct7) {
      is(0x50.U) {
        loopWsConfig.max_k := rs2Data(15, 0)
        loopWsConfig.max_j := rs2Data(31, 16)
        loopWsConfig.max_i := rs2Data(47, 32)
      }
      is(0x51.U)(loopWsConfig.dram_addr_a := rs2Data(38, 0))
      is(0x52.U)(loopWsConfig.dram_addr_b := rs2Data(38, 0))
      is(0x53.U)(loopWsConfig.dram_addr_d := rs2Data(38, 0))
      is(0x54.U)(loopWsConfig.dram_addr_c := rs2Data(38, 0))
      is(0x55.U) {
        loopWsConfig.stride_a := rs2Data(31, 0)
        loopWsConfig.stride_b := rs2Data(63, 32)
      }
      is(0x56.U) {
        loopWsConfig.stride_d := rs2Data(31, 0)
        loopWsConfig.stride_c := rs2Data(63, 32)
      }
    }
  }

  when(io.cmdReq.fire && isLoopConvConfig) {
    configRespValid           := true.B
    configRespBits.rob_id     := io.cmdReq.bits.rob_id
    configRespBits.is_sub     := io.cmdReq.bits.is_sub
    configRespBits.sub_rob_id := io.cmdReq.bits.sub_rob_id
    switch(funct7) {
      is(0x60.U) {
        loopConvConfig.batch_size  := rs2Data(15, 0)
        loopConvConfig.in_dim      := rs2Data(31, 16)
        loopConvConfig.in_channels := rs2Data(47, 32)
      }
      is(0x61.U) {
        loopConvConfig.out_channels := rs2Data(15, 0)
        loopConvConfig.out_dim      := rs2Data(31, 16)
        loopConvConfig.stride       := rs2Data(39, 32)
        loopConvConfig.padding      := rs2Data(47, 40)
      }
      is(0x62.U) {
        loopConvConfig.kernel_dim   := rs2Data(7, 0)
        loopConvConfig.pool_size    := rs2Data(15, 8)
        loopConvConfig.pool_stride  := rs2Data(23, 16)
        loopConvConfig.pool_padding := rs2Data(31, 24)
      }
      is(0x63.U)(loopConvConfig.dram_addr_bias   := rs2Data(38, 0))
      is(0x64.U)(loopConvConfig.dram_addr_input  := rs2Data(38, 0))
      is(0x65.U)(loopConvConfig.dram_addr_weight := rs2Data(38, 0))
      is(0x66.U)(loopConvConfig.dram_addr_output := rs2Data(38, 0))
      is(0x67.U) {
        loopConvConfig.input_stride  := rs2Data(31, 0)
        loopConvConfig.weight_stride := rs2Data(63, 32)
      }
      is(0x68.U)(loopConvConfig.output_stride    := rs2Data(31, 0))
    }
  }

  // =========================================================================
  // Loop trigger: latch bank IDs and start unroller (no cmdResp)
  // Bank values come from rs2Data in the trigger instruction, but loopWsConfig
  // Reg won't update until next edge. Override start.bits combinationally.
  // =========================================================================
  matmulUnroller.io.start.valid := false.B
  matmulUnroller.io.start.bits  := loopWsConfig

  when(io.cmdReq.fire && isLoopWsTrigger) {
    loopWsConfig.bank_a                 := rs2Data(9, 0)
    loopWsConfig.bank_b                 := rs2Data(19, 10)
    loopWsConfig.bank_c                 := rs2Data(29, 20)
    loopWsConfig.low_d                  := rs2Data(30)
    matmulUnroller.io.start.valid       := true.B
    matmulUnroller.io.start.bits.bank_a := rs2Data(9, 0)
    matmulUnroller.io.start.bits.bank_b := rs2Data(19, 10)
    matmulUnroller.io.start.bits.bank_c := rs2Data(29, 20)
    matmulUnroller.io.start.bits.low_d  := rs2Data(30)
  }

  convUnroller.io.start.valid := false.B
  convUnroller.io.start.bits  := loopConvConfig

  when(io.cmdReq.fire && isLoopConvTrigger) {
    loopConvConfig.bank_input              := rs2Data(9, 0)
    loopConvConfig.bank_weight             := rs2Data(19, 10)
    loopConvConfig.bank_output             := rs2Data(29, 20)
    loopConvConfig.no_bias                 := rs2Data(30)
    convUnroller.io.start.valid            := true.B
    convUnroller.io.start.bits.bank_input  := rs2Data(9, 0)
    convUnroller.io.start.bits.bank_weight := rs2Data(19, 10)
    convUnroller.io.start.bits.bank_output := rs2Data(29, 20)
    convUnroller.io.start.bits.no_bias     := rs2Data(30)
  }

  // =========================================================================
  // LoopUnrollers → Arbiter → LoopCmdEncoder → io.subRobReq
  // =========================================================================
  val cmdArb = Module(new Arbiter(new LoopCmd(b), 2))
  cmdArb.io.in(0) <> matmulUnroller.io.cmd
  cmdArb.io.in(1) <> convUnroller.io.cmd
  encoder.io.cmd <> cmdArb.io.out
  encoder.io.subRobRow <> io.subRobReq
  encoder.io.ballId      := ballCommonConfig.ballId.U
  encoder.io.masterRobId := rob_id_reg

  // =========================================================================
  // cmdReq.ready: route to correct consumer
  // =========================================================================
  io.cmdReq.ready := Mux(
    isExUnit,
    exCtrl.exio.cmdReq.ready,
    Mux(
      isLoopWsConfig || isLoopConvConfig,
      true.B,
      Mux(
        isLoopWsTrigger,
        !matmulUnroller.io.busy,
        Mux(isLoopConvTrigger, !convUnroller.io.busy, false.B)
      )
    )
  )

  // =========================================================================
  // cmdResp: mux between exUnit and config immediate response
  // =========================================================================
  io.cmdResp <> exCtrl.exio.cmdResp
  when(configRespValid) {
    io.cmdResp.valid := true.B
    io.cmdResp.bits  := configRespBits
  }

  // =========================================================================
  // Bank connections (unchanged from original)
  // =========================================================================
  for (i <- 0 until inBW) {
    io.bankRead(i).io.req <> exCtrl.exio.bankReadReq(i)
    exCtrl.exio.bankReadResp(i) <> io.bankRead(i).io.resp
    io.bankRead(i).rob_id   := rob_id_reg
    io.bankRead(i).ball_id  := 0.U
    io.bankRead(i).group_id := 0.U
  }
  io.bankRead(0).bank_id := exCtrl.exio.op1_bank_o
  if (inBW > 1) {
    io.bankRead(1).bank_id := exCtrl.exio.op2_bank_o
  }

  for (i <- 0 until outBW) {
    io.bankWrite(i).io <> exCtrl.exio.bankWrite(i)
    io.bankWrite(i).bank_id  := exCtrl.exio.wr_bank_o
    io.bankWrite(i).rob_id   := rob_id_reg
    io.bankWrite(i).ball_id  := 0.U
    io.bankWrite(i).group_id := i.U
  }

  io.status <> exCtrl.exio.status
}
