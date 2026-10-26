package framework.acc.frontend

import framework.core.memory.access.preflight.configs.PmpParams
import framework.acc.frontend.configs.FrontendParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.memdomain.MemoryContext

class CommandPacket(permissionParam: PmpParams, fp: FrontendParam) extends Bundle {
  val commandDataBits = fp.commandDataBits
  val instruction     = new Command(commandDataBits)
  val context         = new MemoryContext(permissionParam)
  val runtime         = Bool()
}

/** Holds an accepted RoCC command until the frontend takes it. */
@instantiable
class Ingress(permissionParam: PmpParams, fp: FrontendParam) extends Module {
  val commandDataBits = fp.commandDataBits

  @public
  val io = IO(new Bundle {
    val rocc    = new RoCCIO(commandDataBits, permissionParam)
    val command = Decoupled(new CommandPacket(permissionParam, fp))
    val idle    = Input(Bool())
    val fault   = Input(Bool())
  })

  val pending = RegInit(false.B)
  val packet  = Reg(new CommandPacket(permissionParam, fp))
  io.rocc.cmd.ready             := !pending && !io.fault && !reset.asBool
  when(io.rocc.cmd.fire) {
    assert(io.rocc.cmd.bits.instruction.raw_inst(6, 0) === "h7b".U, "RoCC accepts accelerator commands only")
    assert(
      !io.rocc.cmd.bits.instruction.raw_inst(14),
      "Accelerator commands report completion without register responses"
    )
    packet.instruction       := io.rocc.cmd.bits.instruction
    packet.context.satp      := io.rocc.cmd.bits.satp
    packet.context.privilege := io.rocc.cmd.bits.effectivePrivilege
    packet.context.sum       := io.rocc.cmd.bits.sum
    packet.context.mxr       := io.rocc.cmd.bits.mxr
    packet.context.pmp       := io.rocc.cmd.bits.pmp
    packet.runtime           := true.B
    pending                  := true.B
  }
  io.command.valid              := pending && !io.fault
  io.command.bits               := packet
  when(io.command.fire)(pending := false.B)
  io.rocc.busy                  := pending || !io.idle || io.fault
  io.rocc.interrupt             := io.fault
  io.rocc.resp.valid            := false.B
  io.rocc.resp.bits             := 0.U.asTypeOf(io.rocc.resp.bits)
}
