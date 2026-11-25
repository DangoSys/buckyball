package framework.generic.acc.shm

import framework.generic.acc.configs.BallLink
import framework.generic.contract.blink.BlinkParam

import framework.generic.acc.shm.BankAccess
import framework.generic.contract.shm.LocalBank

import framework.generic.acc.memdomain.LayoutConfig
import framework.generic.contract.memdomain.MvoverISA
import framework.generic.contract.memdomain.MvoverPort
import framework.generic.acc.memdomain.bank_router.BankMapping
import chisel3._
import chisel3.util._
import framework.generic.acc.frontend.configs.FrontendParam
import framework.generic.acc.memdomain.configs.MemDomainParam
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
