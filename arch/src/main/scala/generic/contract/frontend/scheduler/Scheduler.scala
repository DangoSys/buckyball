package framework.generic.contract.frontend.scheduler

import framework.generic.contract.blink.BlinkParam

import framework.generic.contract.memdomain.toddr.DmaError

import framework.generic.contract.memdomain.MemoryContext

import framework.generic.arbiter.{Arbiter}
import framework.generic.queue.Queue
import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.generic.contract.sims.configs.SimParam
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.frontend.decoder.DomainId
import framework.generic.contract.frontend.decoder.PostGDCmd
import framework.generic.contract.frontend.scheduler.scoreboard.BankAccessInfo
import framework.generic.contract.memdomain.{Response => MemoryResponse}
import framework.generic.contract.frontend.scheduler.Complete
import framework.generic.contract.frontend.scheduler.Issue
import framework.generic.contract.frontend.scheduler.MemoryIssue

/** Work dispatched to a domain, independent of the ROB's storage representation. */
class Issue(link: BlinkParam) extends Bundle {
  val cmd               = new PostGDCmd(link)
  val renamedBankAccess = new BankAccessInfo(link.bankIdBits)
  val rob_id            = UInt(link.robIdBits.W)
  val instId            = UInt(64.W) // Observation identity; never controls scheduling.
  val target            = UInt(5.W)
}

/** Memory dispatch alone carries the submitting address-space state. */
class MemoryIssue(link: BlinkParam, permissionParam: PmpParams) extends Bundle {
  val operation = new Issue(link)
  val context   = new MemoryContext(permissionParam)
}

/** Completion of an admitted operation; legality is an issue-time contract. */
class Complete(robIdBits: Int) extends Bundle {
  val rob_id = UInt(robIdBits.W)
}
