package framework.ball.sub_rob

import framework.generic.contract.ball.BallHostParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.ball.mmio_bank.{MmioRead, MmioWriteReq}
import framework.generic.contract.configs.BallDomainParam
import framework.generic.contract.frontend.scheduler.Complete

/** CISC operands presented to a Ball-owned expander; address-space state stays in BallDomain. */
class ExpansionCommand(host: BallHostParam) extends Bundle {
  val rob_id = UInt(host.link.robIdBits.W)
  val funct  = UInt(7.W)
  val rs2    = UInt(host.link.dataBits.W)
}

/** The host parameters of the SubROB rows: what an expanding Ball is built from besides its link. */
case class ExpansionParam(host: BallHostParam, balls: BallDomainParam)

/** A Ball-owned expander supplies rows to the shared SubROB. */
class Expansion(host: BallHostParam, b: BallDomainParam) extends Bundle {
  val link     = host.link
  val command  = Flipped(Decoupled(new ExpansionCommand(host)))
  val accepts  = Output(Bool())
  val starts   = Output(Bool())
  val rows     = Decoupled(new SubRobRow(host, b))
  val complete = Decoupled(new Complete(host.link.robIdBits))
  val read     = Flipped(new MmioRead(link))
  val write    = Decoupled(new MmioWriteReq(link))
}

@instantiable
trait HasExpansion {
  @public
  val expansion: Expansion
}
