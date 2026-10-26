package framework.acc.gpdomain.decoder

import framework.lib.queue.Queue
import chisel3.experimental.hierarchy.{Instance, Instantiate}

import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.frontend.scheduler.Issue
import framework.acc.memdomain.{BankLayout, BankLayoutRequest}

@instantiable
class Decoder(val frontendParam: FrontendParam, val memParam: MemDomainParam) extends Module {
  val commandDataBits = frontendParam.commandDataBits

  @public
  val io = IO(new Bundle {
    val issue          = Flipped(Decoupled(new Issue(frontendParam, memParam)))
    val layoutRequest  = Decoupled(new BankLayoutRequest(frontendParam))
    val layoutResponse = Flipped(Decoupled(new BankLayout(memParam)))
    val request        = Decoupled(new Issue(frontendParam, memParam))
  })

  val run = io.issue.bits.cmd.cmd.raw_inst(31, 25) === DISA.RUN_KERNEL_BITPAT

  val pending = Instantiate(new Queue(new Issue(frontendParam, memParam), 4))
  io.layoutRequest.valid           := io.issue.valid && run && pending.io.enq.ready
  pending.io.enq.valid             := io.issue.valid && run && io.layoutRequest.ready
  pending.io.enq.bits              := io.issue.bits
  io.issue.ready                   := run && pending.io.enq.ready && io.layoutRequest.ready
  io.layoutRequest.bits.banks(0)   := io.issue.bits.renamedBankAccess.rd_bank_0_id
  io.layoutRequest.bits.banks(1)   := 0.U // Operand 1 names the program buffer, not a data bank.
  io.layoutRequest.bits.banks(2)   := io.issue.bits.renamedBankAccess.wr_bank_id
  io.layoutRequest.bits.enabled(0) := true.B
  io.layoutRequest.bits.enabled(1) := false.B
  io.layoutRequest.bits.enabled(2) := true.B
  io.request.valid                 := pending.io.deq.valid && io.layoutResponse.valid
  io.request.bits                  := pending.io.deq.bits
  io.request.bits.cmd.op1_col      := io.layoutResponse.bits.groups(0)
  io.request.bits.cmd.op2_col      := 0.U
  io.request.bits.cmd.wr_col       := io.layoutResponse.bits.groups(2)
  pending.io.deq.ready             := io.request.ready && io.layoutResponse.valid
  io.layoutResponse.ready          := io.request.ready && pending.io.deq.valid
  when(io.issue.valid)(assert(run, "GPDomain accepts only RUN_KERNEL"))
}
