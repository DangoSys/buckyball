package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import memcore.memory.bank.{Bank, BankSetParams}

@instantiable
class MeshBankNode(p: MeshSharedMemParams, row: Int, col: Int) extends Module {

  @public
  val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new MeshPacket(p)))
    val response = Decoupled(new MeshPacket(p))
  })

  val bank: Instance[Bank] = Instantiate(new Bank(p.bankParams))
  val pending = RegInit(false.B)
  val request = Reg(new MeshPacket(p))

  bank.io.request.valid      := io.request.valid && !pending
  bank.io.request.bits.addr  := io.request.bits.addr(log2Ceil(p.entriesPerBank) - 1, 0)
  bank.io.request.bits.write := io.request.bits.write
  bank.io.request.bits.data  := io.request.bits.data
  bank.io.request.bits.mask  := io.request.bits.mask
  bank.io.request.bits.tag   := io.request.bits.tag
  io.request.ready           := bank.io.request.ready && !pending

  when(io.request.fire) {
    assert(io.request.bits.addr < p.entriesPerBank.U, "Mesh shared request exceeds bank depth")
    assert(io.request.bits.destRow === row.U && io.request.bits.destCol === col.U)
    request := io.request.bits
    pending := true.B
  }

  io.response.valid      := pending && bank.io.response.valid
  io.response.bits       := request
  io.response.bits.tdest := Cat(request.sourceRow, request.sourceCol)
  io.response.bits.tdata := bank.io.response.bits.data
  io.response.bits.tuser := Cat(
    row.U(p.rowBits.W),
    col.U(p.colBits.W),
    request.channel,
    request.addr,
    request.write,
    bank.io.response.bits.error
  )
  bank.io.response.ready := io.response.ready && pending
  when(bank.io.response.valid) {
    assert(pending)
    assert(bank.io.response.bits.tag === request.tag)
  }
  when(io.response.fire) {
    pending := false.B
  }
}
