package examples.designs.poly.harts.poly

import framework.generic.contract.tlink.TaskContext

import framework.thread.acc.memdomain.tomem.configs.{Params => TransportParams}
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.hart.hlink.ServiceData
import framework.generic.contract.controller.{TaskEndpoint}
import framework.thread.tlink.Transport
import framework.generic.dma.Dma
import framework.generic.contract.memory.access.preflight.configs.PmpParams

/** Hart-owned payload DMA. The remote descriptor is opaque to this memory kernel. */
@instantiable
class ServiceTransfer(p: TransportParams, permission: PmpParams) extends Module {

  @public val io = IO(new Bundle {
    val host    = Flipped(new TaskEndpoint)
    val remote  = new TaskEndpoint
    val payload = new ServiceData
    val context = Input(new TaskContext(permission))
    val launch  = Valid(new TaskContext(permission))
    val memory  = new Dma(p.axi)
    val active  = Output(Bool())
  })

  val transport                        = Instantiate(new Transport(p, permission))
  io.memory <> transport.io.link
  val idle :: launch :: running :: Nil = Enum(3)
  val state                            = RegInit(idle)
  val address                          = Reg(UInt(64.W))
  val staged                           = RegInit(false.B)
  val bytes                            = Reg(UInt(32.W))
  val outgoing                         = Reg(Bool())
  val context                          = Reg(new TaskContext(permission))
  val done                             = RegInit(false.B)
  val request                          = io.host.request.bits
  val prepare                          = request.operation === 18.U
  val start                            = request.operation === 19.U
  io.remote.request.valid                     := io.host.request.valid && state === idle
  io.remote.request.bits                      := request
  when(prepare) {
    io.remote.request.bits.context := 0.U
    io.remote.request.bits.field   := 0.U
  }
  io.host.request.ready                       := state === idle && io.remote.request.ready
  io.host.reply.valid                         := io.remote.reply.valid && (state === idle || done)
  io.host.reply.bits                          := io.remote.reply.bits
  io.remote.reply.ready                       := io.host.reply.ready && (state === idle || done)
  io.host.busy                                := io.remote.busy || state =/= idle
  io.active                                   := state =/= idle
  io.launch.valid                             := state === launch
  io.launch.bits                              := context
  when(io.host.request.fire && prepare) {
    assert(!staged, "Payload address already staged")
    address := Cat(request.context, request.field)
    staged  := true.B
  }
  when(io.host.request.fire && start) {
    assert(
      staged && request.context === 0.U && request.field <= 1.U &&
        request.data > 0.U && request.data <= 65536.U,
      "Invalid service payload"
    )
    bytes    := request.data
    outgoing := request.field === 1.U
    context  := io.context
    staged   := false.B
    done     := false.B
    state    := launch
  }
  val dma = transport.io.dma
  dma.read.valid                              := state === launch && outgoing
  dma.read.bits                               := 0.U.asTypeOf(dma.read.bits)
  dma.read.bits.transfer.vaddr                := address
  dma.read.bits.transfer.len                  := bytes
  dma.read.bits.transfer.groups               := 1.U
  dma.read.bits.transfer.stride               := 1.U
  dma.write.valid                             := state === launch && !outgoing
  dma.write.bits                              := 0.U.asTypeOf(dma.write.bits)
  dma.write.bits.vaddr                        := address
  dma.write.bits.len                          := bytes
  when(dma.read.fire || dma.write.fire)(state := running)
  io.payload.send.valid                       := dma.readResult.valid
  // Zero the unused bytes of the last beat before publishing it remotely.
  val tail = bytes(3, 0)
  val mask =
    Mux(dma.readResult.bits.last && tail =/= 0.U, ((1.U(129.W) << (tail << 3)) - 1.U)(127, 0), Fill(128, 1.U(1.W)))
  io.payload.send.bits                                := dma.readResult.bits.data & mask
  dma.readResult.ready                                := io.payload.send.ready
  when(dma.readResult.fire) {
    assert(dma.readResult.bits.fault.error === 0.U, "Service payload read failed")
    when(dma.readResult.bits.last)(done := true.B)
  }
  val remaining = Reg(UInt(32.W))
  when(dma.write.fire)(remaining                      := bytes)
  dma.writeData.valid                                 := io.payload.receive.valid
  dma.writeData.bits.data                             := io.payload.receive.bits
  dma.writeData.bits.last                             := remaining <= 16.U
  io.payload.receive.ready                            := dma.writeData.ready
  when(dma.writeData.fire)(remaining                  := remaining - 16.U)
  dma.writeResult.ready                               := true.B
  when(dma.writeResult.fire) {
    assert(dma.writeResult.bits.fault.error === 0.U, "Service payload write failed")
    done := true.B
  }
  when(io.host.reply.fire && state === running)(state := idle)
}
