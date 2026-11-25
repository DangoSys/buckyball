package framework.generic.contract.frontend.decoder

import framework.generic.contract.blink.BlinkParam

import framework.generic.contract.memdomain.MemoryContext

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.frontend.Command
import framework.generic.contract.frontend.scheduler.scoreboard.BankAccessInfo
import framework.generic.contract.frontend.decoder.PostGDCmd

class PostGDCmd(link: BlinkParam) extends Bundle {
  val commandDataBits = link.dataBits

  val runtime                  = Bool()
  val domain_id                = UInt(4.W)
  val iter                     = UInt(link.iterBits.W)
  val cmd                      = new Command(commandDataBits)
  val bankAccess               = new BankAccessInfo(link.bankIdBits)
  val architecturalDestination = UInt(link.bankIdBits.W)
  val op1_col                  = UInt(link.groupCountBits.W)
  val op2_col                  = UInt(link.groupCountBits.W)
  val wr_col                   = UInt(link.groupCountBits.W)
  val wr_read_bank             = UInt(link.bankIdBits.W)
  val isBarrier                = Bool()
}
