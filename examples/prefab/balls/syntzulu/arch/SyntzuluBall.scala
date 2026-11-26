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
    val bank_mode = Input(Bool())
    val bank_hit = Input(Bool())
    val bank_word = Input(UInt(64.W))
    val weight_request = Output(Bool())
    val weight_address = Output(UInt(7.W))
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
  require(link.dataBits >= 40 && link.bankWidth >= 32 && link.bankEntries >= 8)
  private val operations = Seq("SYNTZULU_WEIGHT", "SYNTZULU_RUN", "SYNTZULU_READ")
  private val byName = route.instructions.map(i => i.mnemonic -> i.funct7).toMap
  private val bankRunCode = byName.get("SYNTZULU_RUN_BANK")
  require(operations.forall(n => byName.contains(n) && (byName(n) >> 4) == 2))
  require(byName.keySet.subsetOf(operations.toSet + "SYNTZULU_RUN_BANK"))
  require(bankRunCode.forall(code => (code >> 4) == 3))

  @public val blink = IO(new BlinkIO(link, route))
  val core = Module(new SyntzuluBridge)
  core.io.clock := clock
  core.io.reset := reset.asBool

  val idle :: fetch :: receive :: launch :: executing :: selectOutput :: collect :: writing :: waiting :: completing :: Nil = Enum(10)
  val state = RegInit(idle)
  val rob = Reg(UInt(link.robIdBits.W))
  val destination = Reg(UInt(link.bankIdBits.W))
  val group = Reg(UInt(link.groupIdBits.W))
  val result = Reg(UInt(32.W))

  val command = blink.cmdReq
  val matches = VecInit(operations.map(name => command.bits.funct === byName(name).U))
  val bankCommand = bankRunCode.map(code => command.bits.funct === code.U).getOrElse(false.B)
  val fromBank = RegInit(false.B)
  val source = Reg(UInt(link.bankIdBits.W))
  val sourceGroup = Reg(UInt(link.groupIdBits.W))
  val frame = Reg(UInt(16.W))
  val delaySteps = Reg(UInt(2.W))
  val pair = RegInit(0.U(3.W))

  val shared = RegInit(false.B)
  val lineValid = RegInit(false.B)
  val pending = RegInit(false.B)
  val lineIndex = Reg(UInt(6.W))
  val lineData = Reg(UInt(128.W))
  val weightRow = core.io.weight_address(6, 1)
  val hit = lineValid && lineIndex === weightRow
  core.io.bank_mode := shared
  core.io.bank_hit := hit
  core.io.bank_word := Mux(core.io.weight_address(0), lineData(127,64), lineData(63,0))
  val needLine = shared && state === executing && core.io.weight_request && !hit
  command.ready := state === idle && core.io.cmd_ready
  core.io.cmd_valid := (state === idle && command.valid && matches.asUInt.orR) ||
    state === launch || state === selectOutput
  core.io.cmd_data := Mux(state === launch, Cat(delaySteps, frame), Mux(state === selectOutput, pair, command.bits.rs2(39, 0)))
  core.io.cmd_op := Mux(state === launch, 1.U, Mux(state === selectOutput, 2.U, OHToUInt(matches.asUInt)))
  core.io.rsp_ready := state === executing || state === collect
  when(command.fire) {
    assert((matches.asUInt.orR || bankCommand) && command.bits.iter === 1.U)
    rob := command.bits.rob_id
    destination := command.bits.wr_bank
    group := command.bits.wr_group
    source := command.bits.op1_bank
    sourceGroup := command.bits.op1_group
    delaySteps := command.bits.rs2(1, 0)
    shared := bankCommand && command.bits.rs2(2)
    lineValid := false.B
    pending := false.B
    frame := command.bits.rs2(31,16)
    fromBank := bankCommand
    pair := 0.U
    state := Mux(bankCommand, Mux(command.bits.rs2(2) || command.bits.rs2(3), launch, fetch), executing)
  }
  val read = blink.bankRead(0)
  read.bank_id := source
  read.group_id := sourceGroup
  read.rob_id := rob
  read.io.req.valid := state === fetch || (needLine && !pending)
  read.io.req.bits.addr := Mux(needLine, weightRow, 0.U)
  read.io.resp.ready := state === receive || pending
  when(read.io.req.fire && state === fetch) { state := receive }
  when(read.io.req.fire && needLine) { pending := true.B; lineIndex := weightRow; lineValid := false.B }
  when(read.io.resp.fire && pending) { lineData := read.io.resp.bits.data; lineValid := true.B; pending := false.B }
  when(read.io.resp.fire && state === receive) { frame := read.io.resp.bits.data(15, 0); state := launch }
  when(state === launch && core.io.cmd_ready) { state := executing }
  when(state === selectOutput && core.io.cmd_ready) { state := collect }
  when(core.io.rsp_valid && core.io.rsp_ready) {
    result := core.io.rsp_data
    state := Mux(state === executing && fromBank, selectOutput, writing)
  }

  val write = blink.bankWrite(0)
  write.bank_id := destination
  write.group_id := group
  write.rob_id := rob
  write.io.req.valid := state === writing
  write.io.req.bits.addr := pair
  write.io.req.bits.data := result
  write.io.req.bits.mask.foreach(_ := true.B)
  // Allow a same-cycle write acknowledgement without feeding request-ready back into BBus.
  write.io.resp.ready := state === writing || state === waiting
  when(write.io.req.fire) { state := waiting }
  when(write.io.resp.fire) {
    assert(state === waiting || write.io.req.fire)
    assert(write.io.resp.bits.ok)
    when(fromBank && pair =/= 7.U) {
      pair := pair + 1.U
      state := selectOutput
    }.otherwise { state := completing }
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
