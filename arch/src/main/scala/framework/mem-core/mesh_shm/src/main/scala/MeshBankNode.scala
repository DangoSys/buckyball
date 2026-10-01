package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import memcore.memory.bank.{Bank, BankSetParams}

class MeshBankNode(p: MeshSharedMemParams, row: Int, col: Int) extends Module {

  val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new MeshPacket(p)))
    val response = Decoupled(new MeshPacket(p))
  })

  val bank    = Module(new Bank(BankSetParams(p.dataBits, 1, p.entriesPerBank, p.tagBits)))
  val pending = RegInit(false.B)
  val request = Reg(new MeshPacket(p))

  bank.io.request.valid      := io.request.valid && !pending
  bank.io.request.bits.addr  := io.request.bits.addr
  bank.io.request.bits.write := io.request.bits.write
  bank.io.request.bits.data  := io.request.bits.data
  bank.io.request.bits.mask  := io.request.bits.mask
  bank.io.request.bits.tag   := io.request.bits.tag
  io.request.ready           := bank.io.request.ready && !pending

  when(io.request.fire) {
    assert(io.request.bits.destRow === row.U && io.request.bits.destCol === col.U)
    request := io.request.bits
    pending := true.B
  }

  io.response.valid          := pending && bank.io.response.valid
  io.response.bits           := request
  io.response.bits.destRow   := request.sourceRow
  io.response.bits.destCol   := request.sourceCol
  io.response.bits.sourceRow := row.U
  io.response.bits.sourceCol := col.U
  io.response.bits.data      := bank.io.response.bits.data
  io.response.bits.error     := bank.io.response.bits.error
  bank.io.response.ready     := io.response.ready && pending
  when(bank.io.response.valid) {
    assert(pending)
    assert(bank.io.response.bits.tag === request.tag)
  }
  when(io.response.fire) {
    pending := false.B
  }
}
