package framework.frontend.boot

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import framework.balldomain.isa.BallISA
import framework.memdomain.boot.MemBoot
import framework.system.core.rocket.{BuckyballCommand, RoCCCommandBB}
import framework.top.GlobalConfig

@instantiable
class BootRom(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {

    val cmd = Decoupled(new Bundle {
      val cmd = new RoCCCommandBB(b.core.xLen)
    })

    val schedulerIdle = Input(Bool())
    val active        = Output(Bool())
  })

  private val ballInitRecords = b.ballDomain.ballIdMappings.map { mapping =>
    BallISA.init(mapping.ballId)
  }

  private val bootRecords =
    MemBoot.initializationCommands(b) ++ ballInitRecords ++ MemBoot.releaseCommands(b) ++
      Seq.empty[BuckyballCommand]

  private val bootFuncts  = VecInit(bootRecords.map(r => r.funct.U(7.W)))
  private val bootRs1Data = VecInit(bootRecords.map(r => r.rs1.U(b.core.xLen.W)))
  private val bootRs2Data = VecInit(bootRecords.map(r => r.rs2.U(b.core.xLen.W)))
  private val bootPcWidth = math.max(1, log2Ceil(bootRecords.length + 1))

  private val active   = RegInit(true.B)
  private val drain    = RegInit(false.B)
  private val waitIdle = RegInit(false.B)
  private val pc       = RegInit(0.U(bootPcWidth.W))

  private val atEnd = pc === bootRecords.length.U(bootPcWidth.W)

  private val current = Wire(new RoCCCommandBB(b.core.xLen))
  current         := 0.U.asTypeOf(new RoCCCommandBB(b.core.xLen))
  current.funct   := Mux(atEnd, 0.U, bootFuncts(pc))
  current.funct3  := BuckyballCommand.Custom3Funct3.U
  current.opcode  := BuckyballCommand.Custom3Opcode.U
  current.rs1Data := Mux(atEnd, 0.U, bootRs1Data(pc))
  current.rs2Data := Mux(atEnd, 0.U, bootRs2Data(pc))

  private val injectValid = active && !drain && !waitIdle && !atEnd

  when(active && !drain && atEnd) {
    drain := true.B
  }
  when(waitIdle && io.schedulerIdle) {
    waitIdle := false.B
  }
  when(drain && io.schedulerIdle) {
    active := false.B
  }
  when(io.cmd.fire) {
    pc       := pc + 1.U
    waitIdle := true.B
  }

  io.cmd.valid    := injectValid
  io.cmd.bits.cmd := current
  io.active       := active
}
