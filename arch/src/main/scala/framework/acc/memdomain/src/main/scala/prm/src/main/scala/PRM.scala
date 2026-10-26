package framework.acc.memdomain.prm

import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.acc.memdomain.prm.Bank
import framework.ip.memory.bank.{SramReadIO, SramWriteIO}
import framework.lib.hash.BankHashMonitor

class PhysicalBankPort(memParam: MemDomainParam) extends Bundle {
  val sramRead  = new SramReadIO(memParam)
  val sramWrite = new SramWriteIO(memParam)
  val clear     = Input(Bool())
  val clearing  = Output(Bool())
  val bind      = Input(Bool())
}

@instantiable
class PRM(val memParam: MemDomainParam, val diffTest: Boolean) extends Module {

  @public
  val io = IO(new Bundle {
    val banks = Vec(memParam.bankNum, new PhysicalBankPort(memParam))

    val hashes =
      if (diffTest) Some(Output(Vec(memParam.bankNum, UInt(32.W))))
      else None

  })

  val banks: Seq[Instance[Bank]] =
    Seq.fill(memParam.bankNum)(Instantiate(
      new Bank(memParam)
    ))

  for (i <- banks.indices) {
    val bank = banks(i)
    val port = io.banks(i)
    bank.io.sramRead <> port.sramRead
    bank.io.sramWrite <> port.sramWrite
    bank.io.clear := port.clear
    port.clearing := bank.io.clearing
    if (diffTest) {
      val monitor: Instance[BankHashMonitor] =
        Instantiate(new BankHashMonitor(memParam))
      monitor.io.bind             := port.bind
      monitor.io.write.valid      := port.sramWrite.req.valid && bank.io.sramWrite.req.ready
      monitor.io.write.bits.addr  := port.sramWrite.req.bits.addr
      monitor.io.write.bits.mask  := port.sramWrite.req.bits.mask
      monitor.io.write.bits.data  := port.sramWrite.req.bits.data
      bank.io.sramWrite.req.valid := port.sramWrite.req.valid && monitor.io.write.ready
      port.sramWrite.req.ready    := bank.io.sramWrite.req.ready && monitor.io.write.ready
      io.hashes.get(i)            := monitor.io.statusHash
    }
  }
}
