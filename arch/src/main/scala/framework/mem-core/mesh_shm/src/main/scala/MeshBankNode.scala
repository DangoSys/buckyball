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

  val bank: Instance[Bank] = Instantiate(new Bank(BankSetParams(p.dataBits, 1, p.entriesPerBank, p.tagBits)))
  val pending = RegInit(false.B)
  val request = Reg(new MeshPacket(p))

  bank.io.request.valid      := io.request.valid && !pending
  bank.io.request.bits.addr  := io.request.bits.addr
  bank.io.request.bits.write := io.request.bits.tuser(0)
  bank.io.request.bits.data  := io.request.bits.tdata
  bank.io.request.bits.mask  := io.request.bits.tkeep
  bank.io.request.bits.tag   := io.request.bits.tid
  io.request.ready           := bank.io.request.ready && !pending

  when(io.request.fire) {
    assert(io.request.bits.destRow === row.U && io.request.bits.destCol === col.U)
    assert(io.request.bits.tlast && !io.request.bits.tuser(2))
    assert(io.request.bits.tuser(1) === false.B)
    request := io.request.bits
    pending := true.B
  }

  io.response.valid          := pending && bank.io.response.valid
  io.response.bits           := request
  io.response.bits.destRow   := request.sourceRow
  io.response.bits.destCol   := request.sourceCol
  io.response.bits.sourceRow := row.U
  io.response.bits.sourceCol := col.U
  io.response.bits.tdata     := bank.io.response.bits.data
  io.response.bits.tkeep     := Mux(request.tuser(0), 0.U, Fill(p.maskBits, 1.U(1.W)))
  io.response.bits.tuser     := Cat(bank.io.response.bits.error, true.B, request.tuser(0))
  bank.io.response.ready     := io.response.ready && pending
  when(bank.io.response.valid) {
    assert(pending)
    assert(bank.io.response.bits.tag === request.tid)
  }
  when(io.response.fire) {
    pending := false.B
  }
}
