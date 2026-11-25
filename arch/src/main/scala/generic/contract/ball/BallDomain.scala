package framework.generic.contract.ball

import framework.generic.contract.blink.{BallResponse, BlinkParam}

import framework.generic.arbiter.{Arbiter}
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.configs.BallDomainParam
import framework.generic.contract.bbus.BallRequest
import framework.generic.contract.mmio_bank.MetadataWrite
import framework.generic.contract.frontend.scheduler.Complete
import framework.generic.contract.frontend.scheduler.Issue
import framework.generic.contract.frontend.scheduler.MemoryIssue
import framework.generic.contract.memdomain.BankLayout
import framework.generic.contract.memdomain.BankLayoutRequest
import framework.generic.contract.memdomain.BankPort
import framework.generic.contract.memdomain.{Response => MemoryResponse}
import framework.generic.contract.ball.DomainMemory
import framework.generic.contract.ball.OperatorBallPort

/** GP operator Ball calls; the GP domain resolves its launch groups before issuing. */
class OperatorBallPort(link: BlinkParam) extends Bundle {
  val request  = Decoupled(new BallRequest(link))
  val response = Flipped(Decoupled(new BallResponse(link)))
}

/** BallDomain owns child identities; MemDomain receives ordinary tagged memory operations. */
class DomainMemory(link: BlinkParam, p: PmpParams) extends Bundle {
  val request  = Decoupled(new MemoryIssue(link, p))
  val response = Flipped(Decoupled(new MemoryResponse(link.robIdBits)))
  val metadata = new MetadataWrite(link.bankWidth, link.mmioBytes)
}
