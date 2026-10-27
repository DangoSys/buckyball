package examples.balls.gemmini

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.balls.gemmini.configs.GemminiBallParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.balldomain.sub_rob.{Expansion, HasExpansion}
import framework.acc.frontend.decoder.DomainId

/** Loop configuration lives in BallDomain's metadata SRAM, not in a second register file. */
@instantiable
class Loop(
  f:       FrontendParam,
  m:       MemDomainParam,
  routing: BallDomainParam,
  id:      Int)
    extends Module
    with HasExpansion {
  val ball = routing.balls.find(_.id == id).get
  val p    = GemminiBallParam.fromProto(ball.parameters.unpack(classOf[buckyball.balls.gemmini.Params]))
  require(m.mmioTotalBytes >= 136)
  def funct(name: String): UInt = ball.instructions.find(_.mnemonic == name).get.funct7.U(7.W)
  val ws                                                                                    = Seq("BOUNDS", "ADDR_A", "ADDR_B", "ADDR_D", "ADDR_C", "STRIDES_AB", "STRIDES_DC")
    .map(n => funct("GEMMINI_LOOP_WS_CONFIG_" + n))
  val conv                                                                                  = (1 to 9).map(n => funct(s"GEMMINI_LOOP_CONV_WS_CONFIG_$n"))
  @public
  val expansion                                                                             = IO(new Expansion(f, m, routing))
  val idle :: storing :: configured :: reading :: receiving :: starting :: streaming :: Nil = Enum(7)
  val state                                                                                 = RegInit(idle)
  val master                                                                                = Reg(UInt(log2Ceil(f.rob_entries).W))
  val operand                                                                               = Reg(UInt(64.W))
  val byte                                                                                  = RegInit(0.U(7.W))
  val base                                                                                  = Reg(UInt(8.W))
  val convolution                                                                           = Reg(Bool())
  val values                                                                                = Reg(Vec(9, UInt(64.W)))
  val fn                                                                                    = expansion.command.bits.funct
  val wsConfig                                                                              = VecInit(ws.map(fn === _))
  val convConfig                                                                            = VecInit(conv.map(fn === _))
  val triggerWs                                                                             = fn === funct("GEMMINI_LOOP_WS")
  val triggerConv                                                                           = fn === funct("GEMMINI_LOOP_CONV_WS")
  expansion.accepts                   := wsConfig.asUInt.orR || convConfig.asUInt.orR || triggerWs || triggerConv
  expansion.starts                    := triggerWs || triggerConv
  expansion.command.ready             := state === idle
  when(expansion.command.fire) {
    master      := expansion.command.bits.rob_id
    operand     := expansion.command.bits.rs2
    byte        := 0.U
    convolution := triggerConv
    when(expansion.starts) {
      base  := Mux(triggerConv, 64.U, 0.U)
      state := reading
    }.otherwise {
      base  := Mux(wsConfig.asUInt.orR, PriorityEncoder(wsConfig) << 3, 64.U + (PriorityEncoder(convConfig) << 3))
      state := storing
    }
  }
  expansion.write.valid               := state === storing
  expansion.write.bits.addr           := base + byte
  expansion.write.bits.data           := (operand >> (byte << 3))(7, 0)
  when(expansion.write.fire) {
    byte                     := byte + 1.U
    when(byte === 7.U)(state := configured)
  }
  expansion.complete.valid            := state === configured
  expansion.complete.bits.rob_id      := master
  when(expansion.complete.fire)(state := idle)
  expansion.read.req.valid            := state === reading
  expansion.read.req.bits.addr        := base + byte
  when(expansion.read.req.fire)(state := receiving)
  expansion.read.resp.ready           := state === receiving
  when(expansion.read.resp.fire) {
    val index = byte(6, 3)
    values(index) := (values(index) >> 8) | (expansion.read.resp.bits.data.pad(64) << 56)
    byte          := byte + 1.U
    state         := Mux(byte === Mux(convolution, 71.U, 55.U), starting, reading)
  }
  val matmul: Instance[LoopMatmulUnroller] = Instantiate(new LoopMatmulUnroller(f, m, p))
  val convolve: Instance[LoopConvUnroller] = Instantiate(new LoopConvUnroller(f, m, p))
  matmul.io.start.valid := state === starting && !convolution
  val mat = matmul.io.start.bits
  mat.max_i               := values(0)(47, 32); mat.max_j   := values(0)(31, 16); mat.max_k := values(0)(15, 0)
  mat.dram_addr_a         := values(1); mat.dram_addr_b     := values(2); mat.dram_addr_d   := values(3); mat.dram_addr_c := values(4)
  mat.stride_a            := values(5)(31, 0); mat.stride_b := values(5)(63, 32)
  mat.stride_d            := values(6)(31, 0); mat.stride_c := values(6)(63, 32)
  mat.bank_a              := operand(9, 0); mat.bank_b      := operand(19, 10); mat.bank_c  := operand(29, 20); mat.low_d := operand(30)
  convolve.io.start.valid := state === starting && convolution
  val cv = convolve.io.start.bits
  cv.batch_size                                              := values(0)(15, 0); cv.in_dim        := values(0)(31, 16); cv.in_channels   := values(0)(47, 32)
  cv.out_channels                                            := values(1)(15, 0); cv.out_dim       := values(1)(31, 16); cv.stride        := values(1)(39, 32);
  cv.padding                                                 := values(1)(47, 40)
  cv.kernel_dim                                              := values(2)(7, 0); cv.pool_size      := values(2)(15, 8); cv.pool_stride    := values(2)(23, 16);
  cv.pool_padding                                            := values(2)(31, 24)
  cv.dram_addr_bias                                          := values(3); cv.dram_addr_input      := values(4); cv.dram_addr_weight      := values(5);
  cv.dram_addr_output                                        := values(6)
  cv.input_stride                                            := values(7)(31, 0); cv.weight_stride := values(7)(63, 32); cv.output_stride := values(8)(31, 0)
  cv.bank_input                                              := operand(9, 0); cv.bank_weight      := operand(19, 10); cv.bank_output     := operand(29, 20);
  cv.no_bias                                                 := operand(30)
  when(matmul.io.start.fire || convolve.io.start.fire)(state := streaming)
  val commands = Wire(Decoupled(new LoopCmd(f, m, p)))
  commands.valid                                              := Mux(convolution, convolve.io.cmd.valid, matmul.io.cmd.valid)
  commands.bits                                               := Mux(convolution, convolve.io.cmd.bits, matmul.io.cmd.bits)
  matmul.io.cmd.ready                                         := !convolution && commands.ready
  convolve.io.cmd.ready                                       := convolution && commands.ready
  expansion.rows.valid                                        := state === streaming && commands.valid
  commands.ready                                              := state === streaming && expansion.rows.ready
  expansion.rows.bits                                         := 0.U.asTypeOf(expansion.rows.bits)
  expansion.rows.bits.last                                    := commands.bits.last
  expansion.rows.bits.ball_id                                 := ball.id.U
  expansion.rows.bits.master_rob_id                           := master
  when(expansion.rows.fire && expansion.rows.bits.last)(state := idle)
  for (i <- 0 until 4) {
    val src     = commands.bits.slots(i)
    val out     = expansion.rows.bits.slots(i)
    val c       = out.cmd
    val t       = src.bits
    val alloc   = t.cmdType === LoopSubCmdType.MSET_ALLOC
    val free    = t.cmdType === LoopSubCmdType.MSET_FREE
    val load    = t.cmdType === LoopSubCmdType.MVIN
    val store   = t.cmdType === LoopSubCmdType.MVOUT
    val preload = t.cmdType === LoopSubCmdType.PRELOAD
    val compute = t.cmdType === LoopSubCmdType.COMPUTE
    val fn      = MuxCase(
      funct("GEMMINI_COMPUTE_ACCUMULATED"),
      Seq(
        (alloc || free)                     -> 32.U,
        load                                -> 33.U,
        store                               -> 16.U,
        preload                             -> funct("GEMMINI_PRELOAD"),
        (compute && t.compute_mode === 0.U) -> funct("GEMMINI_COMPUTE_PRELOADED")
      )
    )
    out.valid := src.valid
    c.domain_id                  := Mux(alloc || free || load || store, DomainId.MEM, DomainId.BALL)
    c.cmd.raw_inst               := (fn << 25) | "h307b".U
    c.cmd.rs1Data                := Mux(
      alloc || free,
      t.bank_id,
      Mux(
        load,
        t.bank_id << 20,
        Mux(store, t.bank_id, t.op1_bank | (t.op2_bank << 10) | (t.wr_bank << 20))
      ) | (t.iter << 30)
    )
    c.cmd.rs2Data                := Mux(
      alloc,
      t.bank_row | (t.bank_col << 5) | (1.U << 10),
      Mux(
        free,
        0.U,
        Mux(
          load || store,
          t.dram_addr | (t.stride.pad(64) << 39),
          Mux(preload, 1.U, Mux(t.compute_mode === 0.U, 2.U, 3.U)) | (t.zero_op2 << 4) | (t.zero_op1_tail << 5)
        )
      )
    )
    c.iter                       := t.iter
    c.bankAccess.rd_bank_0_valid := store || preload || compute
    c.bankAccess.rd_bank_0_id    := Mux(store, t.bank_id, t.op1_bank)
    c.bankAccess.rd_bank_1_valid := compute && !t.zero_op2
    c.bankAccess.rd_bank_1_id    := t.op2_bank
    c.bankAccess.wr_bank_valid   := alloc || free || load || compute
    c.bankAccess.wr_bank_id      := Mux(alloc || free || load, t.bank_id, t.wr_bank)
    c.architecturalDestination   := c.bankAccess.wr_bank_id
    c.bankAccess.rd_wr_valid     := compute && t.compute_mode =/= 0.U
    c.bankAccess.rd_wr_id        := t.wr_bank
    c.wr_read_bank               := t.wr_bank
    c.op1_col                    := 1.U; c.op2_col := 1.U; c.wr_col := 4.U
  }
}
