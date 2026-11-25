package examples.designs.poly.balls.mxint8unpack

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ball.blink.{BlinkIO, HasBlink}
import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.mxint8unpack.{Params => ParametersProto}

@instantiable
class Mxint8UnpackBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball          = route
  ball.parameters.unpack(classOf[ParametersProto])
  require(ball.inBW == 1 && ball.outBW == 1)
  require(ball.instructions.forall(i => (i.funct7 >> 4) == 3))
  require(link.bankWidth == 128 && link.bankMaskLen == 16 && link.bankEntries == 1024)
  require(link.dataBits == 64)
  @public val blink = IO(new BlinkIO(link, route))

  val idle :: metadataRequest :: metadataResponse :: codes :: write :: complete :: Nil = Enum(6)
  val state                                                                            = RegInit(idle)
  val rob                                                                              = Reg(UInt(link.robIdBits.W))
  val inputBank                                                                        = Reg(UInt(link.bankIdBits.W))
  val outputBank                                                                       = Reg(UInt(link.bankIdBits.W))
  val inputGroup                                                                       = Reg(UInt(link.groupIdBits.W))
  val outputGroup                                                                      = Reg(UInt(link.groupIdBits.W))
  val blocks                                                                           = Reg(UInt(8.W))
  val block                                                                            = Reg(UInt(7.W))
  val metadata                                                                         = Reg(UInt(128.W))
  val codeWords                                                                        = Reg(Vec(8, UInt(32.W)))
  val readIssued                                                                       = Reg(UInt(2.W))
  val readReceived                                                                     = Reg(UInt(2.W))
  val writeIssued                                                                      = Reg(UInt(4.W))
  val writeReceived                                                                    = Reg(UInt(4.W))

  blink.cmdReq.ready        := state === idle
  blink.cmdResp.valid       := state === complete
  blink.cmdResp.bits.rob_id := rob
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle && state =/= complete

  val input = blink.bankRead(0)
  input.rob_id           := rob
  input.bank_id          := inputBank
  input.group_id         := inputGroup
  input.io.req.valid     := state === metadataRequest || (state === codes && readIssued < 2.U)
  input.io.req.bits.addr := Mux(state === metadataRequest, (blocks << 1) + (block >> 4), (block << 1) + readIssued)
  input.io.resp.ready    := state === metadataResponse || state === codes

  val exponent = (metadata >> (block(3, 0) << 3))(7, 0)
  val word     = codeWords(writeIssued(2, 0))

  val expanded = (0 until 4).map { lane =>
    val code       = word(8 * lane + 7, 8 * lane)
    val signed     = code.asSInt.pad(9)
    val magnitude  = Mux(code(7), -signed, signed).asUInt
    val highest    = 7.U(3.W) - PriorityEncoder(Reverse(magnitude(7, 0)))
    val field      = exponent +& highest
    val normalized = (magnitude(7, 0) << (7.U(3.W) - highest))(7, 0)
    val normal     = Cat(code(7), field(7, 0), normalized(6, 0), 0.U(16.W))
    val infinity   = Cat(code(7), "h7f800000".U(31.W))
    val subnormal  = Cat(code(7), "h00400000".U(31.W))
    Mux(code === 0.U, 0.U(32.W), Mux(field >= 255.U, infinity, Mux(field === 0.U, subnormal, normal)))
  }

  val output = blink.bankWrite(0)
  output.rob_id           := rob
  output.bank_id          := outputBank
  output.group_id         := outputGroup
  output.io.req.valid     := state === write && writeIssued < 8.U
  output.io.req.bits.addr := (block << 3) + writeIssued
  output.io.req.bits.data := Cat(expanded.reverse)
  output.io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  output.io.resp.ready    := state === write

  when(blink.cmdReq.fire) {
    val cmd = blink.cmdReq.bits
    assert(cmd.rs2 === 0.U, "MXINT8UNPACK reserves rs2")
    assert(cmd.iter > 0.U && cmd.iter <= 128.U, "MXINT8UNPACK requires 1..128 blocks")
    assert(cmd.op1_col === 1.U && cmd.wr_col === 1.U, "MXINT8UNPACK requires single-group banks")
    assert(cmd.op1_bank =/= cmd.wr_bank, "MXINT8UNPACK requires separate physical banks")
    rob        := cmd.rob_id
    inputBank  := cmd.op1_bank; outputBank   := cmd.wr_bank
    inputGroup := cmd.op1_group; outputGroup := cmd.wr_group
    blocks     := cmd.iter; block            := 0.U
    readIssued := 0.U; readReceived          := 0.U
    state      := metadataRequest
  }
  when(state === metadataRequest && input.io.req.fire)(state := metadataResponse)
  when(state === metadataResponse && input.io.resp.fire) {
    metadata   := input.io.resp.bits.data
    readIssued := 0.U; readReceived := 0.U
    state      := codes
  }
  when(state === codes) {
    when(input.io.req.fire)(readIssued := readIssued + 1.U)
    when(input.io.resp.fire) {
      for (i <- 0 until 4)
        codeWords(Cat(readReceived(0), i.U(2.W))) := input.io.resp.bits.data(32 * i + 31, 32 * i)
      readReceived                                := readReceived + 1.U
      when(readReceived === 1.U) {
        writeIssued := 0.U; writeReceived := 0.U
        state       := write
      }
    }
  }
  when(state === write) {
    when(output.io.req.fire)(writeIssued := writeIssued + 1.U)
    when(output.io.resp.fire) {
      writeReceived := writeReceived + 1.U
      when(writeReceived === 7.U) {
        when(block +& 1.U === blocks) {
          state := complete
        }.otherwise {
          block      := block + 1.U
          readIssued := 0.U; readReceived := 0.U
          state      := Mux(block(3, 0) === 15.U, metadataRequest, codes)
        }
      }
    }
  }
  when(state === complete && blink.cmdResp.fire)(state       := idle)
}

object Mxint8UnpackBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new Mxint8UnpackBall(link, route)).blink, None)

}
