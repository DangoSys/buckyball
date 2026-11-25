package framework.generic.contract.frontend

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.generic.contract.sims.configs.SimParam
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.frontend.Command
import framework.generic.contract.configs.BallDomainParam
import framework.generic.contract.frontend.scheduler.Complete
import framework.generic.contract.frontend.scheduler.Issue
import framework.generic.contract.frontend.scheduler.MemoryIssue
import framework.generic.contract.memdomain.{Response => MemoryResponse}

class Command(dataBits: Int) extends Bundle {

  val raw_inst = UInt(32.W)
  val pc       = UInt(dataBits.W)
  val rs1Data  = UInt(dataBits.W)
  val rs2Data  = UInt(dataBits.W)
}
