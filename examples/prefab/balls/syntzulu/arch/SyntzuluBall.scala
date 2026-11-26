package examples.balls.syntzulu

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import chisel3.util._
import framework.ball.bbus.{BallFactory, BallPorts}
import framework.ball.blink.{BlinkIO, HasBlink}
import framework.ball.sub_rob.ExpansionParam
import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import buckyball.balls.syntzulu.{Params => ParametersProto}

class SyntzuluBridge extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Bool())
    val cmd_valid = Input(Bool())
    val cmd_ready = Output(Bool())
    val cmd_op = Input(UInt(2.W))
    val cmd_data = Input(UInt(40.W))
    val rsp_valid = Output(Bool())
    val rsp_ready = Input(Bool())
    val rsp_data = Output(UInt(32.W))
  })
  addResource("/syntzulu/SyntzuluBridge.sv")
  addResource("/syntzulu/SyntzuluCore.sv")
}

@instantiable
class SyntzuluBall(link: BlinkParam, route: BallRouteParam) extends Module with HasBlink {
  route.parameters.unpack(classOf[ParametersProto])
  require(route.inBW == 1 && route.outBW == 1 && route.mmioReadBW == 0 && route.mmioWriteBW == 0)
  require(link.dataBits >= 40 && link.bankWidth >= 32)
  private val operations = Seq("SYNTZULU_WEIGHT", "SYNTZULU_RUN", "SYNTZULU_READ")
  private val byName = route.instructions.map(i => i.mnemonic -> i.funct7).toMap
  require(byName.keySet == operations.toSet && byName.values.forall(code => (code >> 4) == 2))

  @public val blink = IO(new BlinkIO(link, route))
  val core = Module(new SyntzuluBridge)
  core.io.clock := clock
  core.io.reset := reset.asBool

  val idle :: executing :: writing :: waiting :: completing :: Nil = Enum(5)
  val state = RegInit(idle)
  val rob = Reg(UInt(link.robIdBits.W))
  val destination = Reg(UInt(link.bankIdBits.W))
  val group = Reg(UInt(link.groupIdBits.W))
  val result = Reg(UInt(32.W))

  val command = blink.cmdReq
  val matches = VecInit(operations.map(name => command.bits.funct === byName(name).U))
  command.ready := state === idle && core.io.cmd_ready
  core.io.cmd_valid := state === idle && command.valid && matches.asUInt.orR
  core.io.cmd_data := command.bits.rs2(39, 0)
  core.io.cmd_op := OHToUInt(matches.asUInt)
  core.io.rsp_ready := state === executing
  when(command.fire) {
    assert(matches.asUInt.orR && command.bits.iter === 1.U)
    rob := command.bits.rob_id
    destination := command.bits.wr_bank
    group := command.bits.wr_group
    state := executing
  }
  when(core.io.rsp_valid && core.io.rsp_ready) {
    result := core.io.rsp_data
    state := writing
  }

  // The current BBus requires at least one read port. This peripheral only writes.
  val read = blink.bankRead(0)
  read.bank_id := 0.U
  read.group_id := 0.U
  read.rob_id := 0.U
  read.io.req.valid := false.B
  read.io.req.bits.addr := 0.U
  read.io.resp.ready := false.B

  val write = blink.bankWrite(0)
  write.bank_id := destination
  write.group_id := group
  write.rob_id := rob
  write.io.req.valid := state === writing
  write.io.req.bits.addr := 0.U
  write.io.req.bits.data := result
  write.io.req.bits.mask.foreach(_ := true.B)
  // Allow a same-cycle write acknowledgement without feeding request-ready back into BBus.
  write.io.resp.ready := state === writing || state === waiting
  when(write.io.req.fire) { state := waiting }
  when(write.io.resp.fire) {
    assert(state === waiting || write.io.req.fire)
    assert(write.io.resp.bits.ok)
    state := completing
  }
  blink.cmdResp.valid := state === completing
  blink.cmdResp.bits.rob_id := rob
  when(blink.cmdResp.fire) { state := idle }
  blink.status.idle := state === idle
  blink.status.running := state =/= idle
}

object SyntzuluBall extends BallFactory {
  def instantiate(link: BlinkParam, route: BallRouteParam, expansion: Option[ExpansionParam]): BallPorts =
    BallPorts(Instantiate(new SyntzuluBall(link, route)).blink, None)
}
