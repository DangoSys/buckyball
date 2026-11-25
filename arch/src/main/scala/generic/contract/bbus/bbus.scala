package framework.generic.contract.bbus

import framework.generic.contract.blink.BlinkParam

import framework.generic.contract.blink.BallResponse

import framework.generic.queue.Queue
import framework.generic.arbiter.{Arbiter, RRArbiter}
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.generic.contract.decoder.DISA.BALL_INIT_BITPAT
import framework.generic.contract.configs.BallDomainParam
import framework.generic.contract.memdomain.BankLayout
import framework.generic.contract.memdomain.BankLayoutRequest
import framework.generic.contract.memdomain.BankPort
import framework.generic.contract.blink.{BallCommand, BlinkParam}
import framework.generic.contract.bbus.BallRequest

/** A Ball command routed to `target`; `enabled` marks which of op1, op2 and wr name a bank. */
class BallRequest(link: BlinkParam) extends Bundle {
  val target  = UInt(5.W)
  val instId  = UInt(64.W) // Observation identity; never controls scheduling.
  val enabled = Vec(3, Bool())
  val cmd     = new BallCommand(link)
}
