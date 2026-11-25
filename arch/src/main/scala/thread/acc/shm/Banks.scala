package framework.thread.acc.shm

import framework.thread.acc.configs.BallLink

import framework.generic.contract.blink.BlinkParam

import framework.thread.acc.shm.BankAccess
import framework.generic.contract.shm.LocalBank

import framework.thread.acc.memdomain.LayoutConfig
import framework.generic.contract.memdomain.MvoverISA
import framework.generic.contract.memdomain.MvoverPort
import framework.thread.acc.memdomain.bank_router.BankMapping
import chisel3._
import chisel3.util._
import framework.thread.acc.frontend.configs.FrontendParam
import framework.thread.acc.memdomain.configs.MemDomainParam
import framework.generic.contract.memdomain.BankPort

/** Hart-local shared-bank access, lifetime and synchronization. */
class BankAccess(frontendParam: FrontendParam, memParam: MemDomainParam) extends Bundle {
  val sharedPorts = memParam.sharedPorts
  require(sharedPorts >= 0 && (memParam.sharedEnable || sharedPorts == 0))
  val requests    = Vec(sharedPorts, new BankPort(BallLink.blink(frontendParam, memParam)))

  val local = Flipped(new LocalBank(
    MvoverISA.AddressBits,
    MvoverISA.BankBits,
    memParam.bankWidth,
    math.max(1, log2Ceil(frontendParam.rob_entries))
  ))

  val config  = Decoupled(new LayoutConfig(frontendParam, memParam))
  val mapping = Decoupled(new BankMapping(frontendParam))
}
