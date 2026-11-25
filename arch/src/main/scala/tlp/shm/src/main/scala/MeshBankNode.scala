package framework.tlp.shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.generic.memory.bank.Bank
import framework.generic.arbiter.RRArbiter
import framework.generic.queue.Queue

/** A mesh node can contain several independently progressing SRAM banks. */
@instantiable
class MeshBankNode(p: MeshSharedMemParams, row: Int, col: Int) extends Module {

  @public val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new MeshPacket(p)))
    val response = Decoupled(new MeshPacket(p))
  })

  val banks    = Seq.fill(p.banksPerNode)(Instantiate(new Bank(p.bankParams)))
  val requests = Seq.fill(p.banksPerNode)(Instantiate(new Queue(new MeshPacket(p), 4)))
  val replies  = Instantiate(new RRArbiter(new MeshPacket(p), p.banksPerNode))
  io.response <> replies.io.out
  io.request.ready := VecInit(banks.zipWithIndex.map { case (bank, i) =>
    bank.io.request.ready && requests(i).io.enq.ready
  })(
    io.request.bits.nodeBank
  )
  for ((bank, i) <- banks.zipWithIndex) {
    bank.io.request.valid      := io.request.valid && io.request.bits.nodeBank === i.U && requests(i).io.enq.ready
    bank.io.request.bits.addr  := io.request.bits.addr(log2Ceil(p.entriesPerBank) - 1, 0)
    bank.io.request.bits.write := io.request.bits.write
    bank.io.request.bits.data  := io.request.bits.data
    bank.io.request.bits.mask  := io.request.bits.mask
    bank.io.request.bits.tag   := io.request.bits.tag
    requests(i).io.enq.valid   := bank.io.request.fire
    requests(i).io.enq.bits    := io.request.bits
    val request  = requests(i).io.deq.bits
    val response = replies.io.in(i)
    response.valid           := requests(i).io.deq.valid && bank.io.response.valid
    response.bits            := request
    response.bits.tdest      := Cat(request.sourceRow, request.sourceCol)
    response.bits.tdata      := bank.io.response.bits.data
    response.bits.tuser      := Cat(
      row.U(p.rowBits.W),
      col.U(p.colBits.W),
      request.channel,
      request.addr,
      request.write,
      bank.io.response.bits.error
    )
    bank.io.response.ready   := response.ready && requests(i).io.deq.valid
    when(response.valid) {
      assert(bank.io.response.bits.tag === request.tag, "Hart storage response metadata is reordered")
    }
    requests(i).io.deq.ready := response.fire
  }
  when(io.request.fire) {
    assert(io.request.bits.addr < p.entriesPerBank.U, "Hart storage request exceeds bank depth")
    assert(io.request.bits.destRow === row.U && io.request.bits.destCol === col.U)
  }
}
