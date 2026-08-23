package examples.balls.gemmini

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.MmioRead
import framework.balldomain.rs.BallRsComplete
import framework.top.GlobalConfig

@instantiable
class GemminiBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {
  require(
    b.frontend.sub_rob_enable,
    "GemminiBall requires frontend.sub_rob_enable=true because loop instructions emit SubROB rows"
  )

  val ballCommonConfig = b.ballDomain.ballIdMappings
    .find(_.ballName == "GemminiBall")
    .getOrElse(
      throw new IllegalArgumentException("GemminiBall not found in config")
    )

  val inBW  = ballCommonConfig.inBW
  val outBW = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink:  BlinkIO    = io
  def status: BallStatus = io.status
  dontTouch(io)

  // =========================================================================
  // Sub-modules
  // =========================================================================
  val exCtrl: Instance[GemminiExCtrl] = Instantiate(new GemminiExCtrl(b))

  val matmulUnroller: Instance[LoopMatmulUnroller] = Instantiate(
    new LoopMatmulUnroller(b)
  )

  val convUnroller: Instance[LoopConvUnroller] = Instantiate(
    new LoopConvUnroller(b)
  )

  val encoder: Instance[LoopCmdEncoder] = Instantiate(new LoopCmdEncoder(b))

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

  def ballFunct(mnemonic: String): UInt =
    b.ballDomain.ballISA
      .find(_.mnemonic == mnemonic)
      .map(_.funct7.U(7.W))
      .getOrElse(throw new IllegalArgumentException(s"$mnemonic not found in ballISA"))

  val configFunct     = ballFunct("GEMMINI_CONFIG")
  val preloadFunct    = ballFunct("GEMMINI_PRELOAD")
  val computePreFunct = ballFunct("GEMMINI_COMPUTE_PRELOADED")
  val computeAccFunct = ballFunct("GEMMINI_COMPUTE_ACCUMULATED")
  val flushFunct      = ballFunct("GEMMINI_FLUSH")

  val loopWsConfigFuncts = Seq(
    ballFunct("GEMMINI_LOOP_WS_CONFIG_BOUNDS"),
    ballFunct("GEMMINI_LOOP_WS_CONFIG_ADDR_A"),
    ballFunct("GEMMINI_LOOP_WS_CONFIG_ADDR_B"),
    ballFunct("GEMMINI_LOOP_WS_CONFIG_ADDR_D"),
    ballFunct("GEMMINI_LOOP_WS_CONFIG_ADDR_C"),
    ballFunct("GEMMINI_LOOP_WS_CONFIG_STRIDES_AB"),
    ballFunct("GEMMINI_LOOP_WS_CONFIG_STRIDES_DC")
  )

  val loopWsTriggerFunct = ballFunct("GEMMINI_LOOP_WS")

  val loopConvConfigFuncts = Seq(
    ballFunct("GEMMINI_LOOP_CONV_WS_CONFIG_1"),
    ballFunct("GEMMINI_LOOP_CONV_WS_CONFIG_2"),
    ballFunct("GEMMINI_LOOP_CONV_WS_CONFIG_3"),
    ballFunct("GEMMINI_LOOP_CONV_WS_CONFIG_4"),
    ballFunct("GEMMINI_LOOP_CONV_WS_CONFIG_5"),
    ballFunct("GEMMINI_LOOP_CONV_WS_CONFIG_6"),
    ballFunct("GEMMINI_LOOP_CONV_WS_CONFIG_7"),
    ballFunct("GEMMINI_LOOP_CONV_WS_CONFIG_8"),
    ballFunct("GEMMINI_LOOP_CONV_WS_CONFIG_9")
  )

  val loopConvTriggerFunct = ballFunct("GEMMINI_LOOP_CONV_WS")

  val rs2Data = io.cmdReq.bits.cmd.special

  val isConfig     = funct7 === configFunct
  val isPreload    = funct7 === preloadFunct
  val isComputePre = funct7 === computePreFunct
  val isComputeAcc = funct7 === computeAccFunct
  val isFlush      = funct7 === flushFunct
  val isExUnit     =
    isConfig || isPreload || isComputePre || isComputeAcc || isFlush

  val isLoopWsConfig    = loopWsConfigFuncts.map(funct7 === _).reduce(_ || _)
  val isLoopWsTrigger   = funct7 === loopWsTriggerFunct
  val isLoopConvConfig  = loopConvConfigFuncts.map(funct7 === _).reduce(_ || _)
  val isLoopConvTrigger = funct7 === loopConvTriggerFunct

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
      is(loopWsConfigFuncts(0)) {
        loopWsConfig.max_k := rs2Data(15, 0)
        loopWsConfig.max_j := rs2Data(31, 16)
        loopWsConfig.max_i := rs2Data(47, 32)
      }
      is(loopWsConfigFuncts(1))(loopWsConfig.dram_addr_a := rs2Data(38, 0))
      is(loopWsConfigFuncts(2))(loopWsConfig.dram_addr_b := rs2Data(38, 0))
      is(loopWsConfigFuncts(3))(loopWsConfig.dram_addr_d := rs2Data(38, 0))
      is(loopWsConfigFuncts(4))(loopWsConfig.dram_addr_c := rs2Data(38, 0))
      is(loopWsConfigFuncts(5)) {
        loopWsConfig.stride_a := rs2Data(31, 0)
        loopWsConfig.stride_b := rs2Data(63, 32)
      }
      is(loopWsConfigFuncts(6)) {
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
      is(loopConvConfigFuncts(0)) {
        loopConvConfig.batch_size  := rs2Data(15, 0)
        loopConvConfig.in_dim      := rs2Data(31, 16)
        loopConvConfig.in_channels := rs2Data(47, 32)
      }
      is(loopConvConfigFuncts(1)) {
        loopConvConfig.out_channels := rs2Data(15, 0)
        loopConvConfig.out_dim      := rs2Data(31, 16)
        loopConvConfig.stride       := rs2Data(39, 32)
        loopConvConfig.padding      := rs2Data(47, 40)
      }
      is(loopConvConfigFuncts(2)) {
        loopConvConfig.kernel_dim   := rs2Data(7, 0)
        loopConvConfig.pool_size    := rs2Data(15, 8)
        loopConvConfig.pool_stride  := rs2Data(23, 16)
        loopConvConfig.pool_padding := rs2Data(31, 24)
      }
      is(loopConvConfigFuncts(3))(loopConvConfig.dram_addr_bias   := rs2Data(38, 0))
      is(loopConvConfigFuncts(4))(loopConvConfig.dram_addr_input  := rs2Data(38, 0))
      is(loopConvConfigFuncts(5))(loopConvConfig.dram_addr_weight := rs2Data(38, 0))
      is(loopConvConfigFuncts(6))(loopConvConfig.dram_addr_output := rs2Data(38, 0))
      is(loopConvConfigFuncts(7)) {
        loopConvConfig.input_stride  := rs2Data(31, 0)
        loopConvConfig.weight_stride := rs2Data(63, 32)
      }
      is(loopConvConfigFuncts(8))(loopConvConfig.output_stride    := rs2Data(31, 0))
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
  val idleReady =
    exCtrl.exio.cmdReq.ready && !matmulUnroller.io.busy && !convUnroller.io.busy
  io.cmdReq.ready := Mux(
    isExUnit,
    exCtrl.exio.cmdReq.ready,
    Mux(
      isLoopWsConfig || isLoopConvConfig,
      idleReady,
      Mux(
        isLoopWsTrigger,
        !matmulUnroller.io.busy,
        Mux(isLoopConvTrigger, !convUnroller.io.busy, idleReady)
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

  MmioRead.tieOff(io.mmioRead)
}
