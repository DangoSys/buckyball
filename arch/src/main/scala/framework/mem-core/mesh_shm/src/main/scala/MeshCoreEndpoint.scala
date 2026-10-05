package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/** The private Bank port stays beside its Core; only packets cross the Mesh. */
@instantiable
class MeshCoreEndpoint(
  p:    MeshSharedMemParams,
  core: Int,
  row:  Int,
  col:  Int)
    extends Module {

  @public
  val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new MeshPacket(p)))
    val response = Decoupled(new MeshPacket(p))
    val bank     = new MeshLocalBankPort(p.addressBits, p.localBankBits, p.dataBits, p.tagBits)
  })

  val pending = RegInit(false.B)
  val request = Reg(new MeshPacket(p))
  io.bank.request.valid          := io.request.valid && !pending
  io.bank.request.bits.bank      := io.request.bits.localBank
  io.bank.request.bits.addr      := io.request.bits.addr
  io.bank.request.bits.write     := io.request.bits.write
  io.bank.request.bits.data      := io.request.bits.data
  io.bank.request.bits.mask      := io.request.bits.mask
  io.bank.request.bits.tag       := io.request.bits.tag
  io.request.ready               := io.bank.request.ready && !pending
  when(io.request.fire) {
    assert(io.request.bits.privateRequest && io.request.bits.core === core.U)
    assert(io.request.bits.destRow === row.U && io.request.bits.destCol === col.U)
    request := io.request.bits
    pending := true.B
  }
  io.response.valid              := pending && io.bank.response.valid
  io.response.bits               := request
  io.response.bits.tdest         := Cat(request.sourceRow, request.sourceCol)
  io.response.bits.tdata         := Mux(request.write, 0.U, io.bank.response.bits.data)
  io.response.bits.tuser         := Cat(
    request.core,
    request.localBank,
    true.B,
    row.U(p.rowBits.W),
    col.U(p.colBits.W),
    request.channel,
    request.addr,
    request.write,
    io.bank.response.bits.error
  )
  io.bank.response.ready         := pending && io.response.ready
  when(io.bank.response.valid) {
    assert(pending && io.bank.response.bits.tag === request.tag)
  }
  when(io.response.fire)(pending := false.B)
}
