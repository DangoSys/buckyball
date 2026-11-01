package framework.core.shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ip.memory.bank.Bank
import framework.lib.arbiter.RRArbiter

/** A mesh node can contain several independently progressing SRAM banks. */
@instantiable
class MeshBankNode(p: MeshSharedMemParams, row: Int, col: Int) extends Module {

  @public val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new MeshPacket(p)))
    val response = Decoupled(new MeshPacket(p))
  })

  val banks    = Seq.fill(p.banksPerNode)(Instantiate(new Bank(p.bankParams)))
  val pending  = RegInit(VecInit(Seq.fill(p.banksPerNode)(false.B)))
  val requests = Reg(Vec(p.banksPerNode, new MeshPacket(p)))
  val replies  = Instantiate(new RRArbiter(new MeshPacket(p), p.banksPerNode))
  io.response <> replies.io.out
  io.request.ready := VecInit(banks.zipWithIndex.map { case (bank, i) => bank.io.request.ready && !pending(i) })(
    io.request.bits.nodeBank
  )
  for ((bank, i) <- banks.zipWithIndex) {
    bank.io.request.valid      := io.request.valid && io.request.bits.nodeBank === i.U && !pending(i)
    bank.io.request.bits.addr  := io.request.bits.addr(log2Ceil(p.entriesPerBank) - 1, 0)
    bank.io.request.bits.write := io.request.bits.write
    bank.io.request.bits.data  := io.request.bits.data
    bank.io.request.bits.mask  := io.request.bits.mask
    bank.io.request.bits.tag   := io.request.bits.tag
    when(bank.io.request.fire) { requests(i) := io.request.bits; pending(i) := true.B }
    val request = requests(i)
    val response = replies.io.in(i)
    response.valid                 := pending(i) && bank.io.response.valid
    response.bits                  := request
    response.bits.tdest            := Cat(request.sourceRow, request.sourceCol)
    response.bits.tdata            := bank.io.response.bits.data
    response.bits.tuser            := Cat(
      row.U(p.rowBits.W),
      col.U(p.colBits.W),
      request.channel,
      request.addr,
      request.write,
      bank.io.response.bits.error
    )
    bank.io.response.ready         := response.ready && pending(i)
    when(bank.io.response.valid) {
      assert(pending(i) && bank.io.response.bits.tag === request.tag)
    }
    when(response.fire)(pending(i) := false.B)
  }
  when(io.request.fire) {
    assert(io.request.bits.addr < p.entriesPerBank.U, "Core storage request exceeds bank depth")
    assert(io.request.bits.destRow === row.U && io.request.bits.destCol === col.U)
  }
}
