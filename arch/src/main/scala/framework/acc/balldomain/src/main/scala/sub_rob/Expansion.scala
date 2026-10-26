package framework.acc.balldomain.sub_rob

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam
import framework.acc.frontend.scheduler.Complete
import framework.acc.balldomain.mmio_bank.{MmioRead, MmioWriteReq}

/** CISC operands presented to a Ball-owned expander; address-space state stays in BallDomain. */
class ExpansionCommand(f: FrontendParam) extends Bundle {
  val rob_id = UInt(log2Up(f.rob_entries).W)
  val funct  = UInt(7.W)
  val rs2    = UInt(f.commandDataBits.W)
}

/** A Ball-owned expander supplies rows to the shared SubROB. */
class Expansion(f: FrontendParam, m: MemDomainParam, b: BallDomainParam) extends Bundle {
  val command  = Flipped(Decoupled(new ExpansionCommand(f)))
  val accepts  = Output(Bool())
  val starts   = Output(Bool())
  val rows     = Decoupled(new SubRobRow(f, m, b))
  val complete = Decoupled(new Complete(f))
  val read     = Flipped(new MmioRead(m))
  val write    = Decoupled(new MmioWriteReq(m))
}

@instantiable
trait HasExpansion {
  @public
  val expansion: Expansion
}
