package framework.system.core.accelerator

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.ant
import framework.system.core.rocket.CpuParams
import hier.core.rocket.{AdmissionPorts, CommandSnapshot}
import memcore.bus.chi
import memcore.bus.chi.rnf.{CacheAccess, CacheResult}
import memcore.memory.interlock.{Acknowledgement, CpuQuery, Maintenance, Params => TrackingParams}

/** Controller-side cache services; requests retain their original admission tag. */
class AntMemoryPort(tracking: TrackingParams, bus: chi.Params) extends Bundle {
  val cpuQuery      = Input(new CpuQuery(tracking))
  val cpuAllow      = Output(Bool())
  val cpuProbeAllow = Output(Bool())
  val maintenance   = Decoupled(new Maintenance(tracking))
  val maintained    = Flipped(Decoupled(new Acknowledgement(tracking)))
  val pteRequest    = Decoupled(new CacheAccess(bus))
  val pteResponse   = Flipped(Decoupled(new CacheResult))
}

/** Binds local Ant execution to the controller's immutable task address-space snapshot. */
@instantiable
class AntAdmission(
  p:                      ant.Params,
  tracking:               TrackingParams,
  pmps:                   Int,
  bus:                    chi.Params
)(
  implicit val cpuParams: CpuParams)
    extends Module {
  require(tracking.entries == 4)

  @public val io = IO(new Bundle {

    val bind = Flipped(Valid(new Bundle {
      val task     = UInt(p.taskBits.W)
      val snapshot = new CommandSnapshot(tracking, pmps)
    }))

    val command     = Flipped(Decoupled(new ant.Command(p)))
    val response    = Decoupled(new ant.Response(p))
    val cancel      = Input(Bool())
    val workDrained = Input(Bool())
    val halted      = Input(Bool())
    val drained     = Output(Bool())
    val admission   = new AdmissionPorts(tracking, pmps, bus)
    val memory      = new AntMemoryPort(tracking, bus)
  })

  val bound             = RegInit(false.B)
  val task              = Reg(UInt(p.taskBits.W))
  val context           = Reg(new CommandSnapshot(tracking, pmps))
  val live              = RegInit(VecInit(Seq.fill(tracking.entries)(false.B)))
  val sent              = RegInit(VecInit(Seq.fill(tracking.entries)(false.B)))
  val pending           = RegInit(false.B)
  val packet            = Reg(new CommandSnapshot(tracking, pmps))
  val responsePending   = RegInit(false.B)
  val responseRd        = Reg(UInt(5.W))
  val responseDelivered = RegInit(false.B)
  val free              = VecInit(live.map(!_)).asUInt
  val selected          = PriorityEncoder(free)
  val port              = io.admission
  io.drained                               := !live.asUInt.orR && !pending && !responsePending && io.workDrained
  when(io.bind.valid) {
    assert(io.drained && !io.command.valid, "Ant context rebound before previous task drained")
    bound   := true.B
    task    := io.bind.bits.task
    context := io.bind.bits.snapshot
  }
  val capacity = bound && free.orR && !pending && !responsePending && !io.cancel && !io.bind.valid && !reset.asBool
  port.reserve.valid                       := io.command.valid && capacity
  port.reserve.bits.id                     := selected
  io.command.ready                         := capacity && port.reserve.ready
  port.command.valid                       := pending
  port.command.bits                        := packet
  when(io.command.fire) {
    val instruction = io.command.bits.instruction
    assert(io.command.bits.task === task, "Ant command uses another task context")
    assert(instruction(6, 0) === "h7b".U, "Ant admission only accepts NPU commands")
    packet                      := context
    packet.tag                  := selected
    packet.instruction.raw_inst := instruction
    packet.instruction.pc       := io.command.bits.pc
    packet.instruction.funct    := instruction(31, 25)
    packet.instruction.funct3   := instruction(14, 12)
    packet.instruction.rs2      := instruction(24, 20)
    packet.instruction.rs1      := instruction(19, 15)
    packet.instruction.xd       := instruction(14)
    packet.instruction.xs1      := instruction(13)
    packet.instruction.xs2      := instruction(12)
    packet.instruction.rd       := instruction(11, 7)
    packet.instruction.opcode   := instruction(6, 0)
    packet.instruction.rs1Data  := io.command.bits.rs1
    packet.instruction.rs2Data  := io.command.bits.rs2
    live(selected)              := true.B
    sent(selected)              := false.B
    pending                     := true.B
    when(instruction(14)) {
      responsePending   := true.B
      responseDelivered := false.B
      responseRd        := instruction(11, 7)
    }
  }
  when(port.command.fire) {
    pending                                       := false.B
    sent(packet.tag(1, 0))                        := true.B
    when(packet.instruction.xd)(responseDelivered := true.B)
  }
  port.complete.ready                      := true.B
  when(port.complete.fire) {
    val tag = port.complete.bits.tag
    assert(
      tag < tracking.entries.U && live(tag(1, 0)) && sent(tag(1, 0)),
      "Ant completion has no delivered live command"
    )
    live(tag(1, 0)) := false.B
    sent(tag(1, 0)) := false.B
  }
  port.cancelled.ready                     := true.B
  assert(!port.cancelled.valid, "Ant cancellation must drain accepted NPU commands")
  assert(!io.halted && !port.interrupt, "Ant NPU backend fault")
  port.outstanding                         := PopCount(live)
  io.response.valid                        := port.response.valid && responsePending
  io.response.bits.task                    := task
  io.response.bits.rd                      := port.response.bits.rd
  io.response.bits.data                    := port.response.bits.data
  io.response.bits.error                   := false.B
  port.response.ready                      := io.response.ready && responsePending
  when(port.response.valid) {
    assert(
      responsePending && (responseDelivered || (port.command.fire && packet.instruction.xd)),
      "Ant received an unsolicited NPU response"
    )
    assert(port.response.bits.rd === responseRd, "Ant NPU response destination changed")
  }
  when(port.response.fire)(responsePending := false.B)
  port.cpuQuery                            := io.memory.cpuQuery
  io.memory.cpuAllow                       := port.cpuAllow
  io.memory.cpuProbeAllow                  := port.cpuProbeAllow
  io.memory.maintenance <> port.maintenance
  port.maintained <> io.memory.maintained
  io.memory.pteRequest <> port.pteRequest
  port.pteResponse <> io.memory.pteResponse
}
