package framework.thread.acc.memdomain.bank_router

import framework.thread.acc.shm.BankAccess

import framework.thread.acc.memdomain.bank_router.BankMapping

import framework.generic.queue.Queue
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.thread.acc.frontend.configs.FrontendParam
import framework.thread.acc.gpdomain.configs.GpDomainParam
import framework.thread.acc.memdomain.configs.MemDomainParam
import framework.generic.contract.memdomain.BankLayout
import framework.generic.contract.memdomain.BankLayoutRequest
import framework.generic.contract.memdomain.BankPort

/** Allocate a renamed binding, or commit a new binding and retire its previous one. */
class BankMapping(frontendParam: FrontendParam) extends Bundle {
  val bank     = UInt(frontendParam.bank_id_len.W)
  val vbank    = UInt(frontendParam.bank_id_len.W)
  val previous = UInt(frontendParam.bank_id_len.W)
  val threadId = UInt(64.W)
  val allocate = Bool()
  val reuse    = Bool()
  val retire   = Bool()
  val write    = Bool()
}
