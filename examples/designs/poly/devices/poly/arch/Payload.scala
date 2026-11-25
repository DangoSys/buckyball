package examples.designs.poly.devices.poly

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.hart.hlink.ServiceData
import framework.generic.contract.controller.{TaskEndpoint}
import framework.device.shm.StorageAccess
import framework.generic.queue.Queue

class PayloadFragment extends Bundle {
  val upper = Bool()
  val half  = Bool()
  val high  = Bool()
  val last  = Bool()
}

/** Device-owned payload stream. SRAM requests and acknowledgements overlap. */
@instantiable
class Payload extends Module {

  @public val io = IO(new Bundle {
    val host    = Flipped(new TaskEndpoint)
    val storage = new TaskEndpoint
    val payload = Flipped(new ServiceData)
    val local   = new StorageAccess(128)
  })

  val idle :: passthrough :: reply :: active :: Nil = Enum(4)
  val state                                         = RegInit(idle)
  val offset                                        = Reg(UInt(64.W))
  val staged                                        = RegInit(false.B)
  val remaining                                     = Reg(UInt(32.W))
  val outgoing                                      = Reg(Bool())
  val upper                                         = RegInit(false.B)
  val pending                                       = RegInit(0.U(3.W))
  val low                                           = Reg(UInt(64.W))
  val inputs                                        = Instantiate(new Queue(UInt(128.W), 4))
  val outputs                                       = Instantiate(new Queue(UInt(128.W), 4))
  val fragments                                     = Instantiate(new Queue(new PayloadFragment, 4))
  val request                                       = io.host.request.bits
  val special                                       = request.operation === 18.U || request.operation === 19.U
  val writing                                       = state === active && outgoing
  val reading                                       = state === active && !outgoing
  io.host.request.ready    := state === idle && (special || io.storage.request.ready)
  io.host.reply.valid      := state === reply || (state === passthrough && io.storage.reply.valid)
  io.host.reply.bits       := Mux(state === passthrough, io.storage.reply.bits, 0.U)
  io.host.busy             := state =/= idle || io.storage.busy
  io.storage.request.valid := state === idle && io.host.request.valid && !special
  io.storage.request.bits  := request
  io.storage.reply.ready   := state === passthrough && io.host.reply.ready
  when(io.host.request.fire) {
    state := passthrough
    when(request.operation === 18.U) {
      assert(!staged && request.context === 0.U && request.field === 0.U && request.data(2, 0) === 0.U)
      offset := request.data
      staged := true.B
      state  := reply
    }
    when(request.operation === 19.U) {
      assert(staged && request.context === 0.U && request.field <= 1.U &&
        request.data > 0.U && request.data <= 65536.U)
      remaining := request.data
      outgoing  := request.field === 1.U
      staged    := false.B
      upper     := false.B
      state     := active
    }
  }
  inputs.io.enq.valid      := writing && io.payload.send.valid
  inputs.io.enq.bits       := io.payload.send.bits
  io.payload.send.ready    := writing && inputs.io.enq.ready
  io.payload.receive.valid := reading && outputs.io.deq.valid
  io.payload.receive.bits  := outputs.io.deq.bits
  outputs.io.deq.ready     := reading && io.payload.receive.ready
  val half   = upper || offset(3) || remaining <= 8.U
  val amount = Mux(half, 8.U, 16.U)
  val last   = remaining <= amount
  val finish = upper || !half || last
  val word   = Mux(upper, Cat(0.U(64.W), inputs.io.deq.bits(127, 64)), inputs.io.deq.bits)
  io.local.request.valid        := state === active && remaining > 0.U && pending < 4.U &&
    Mux(outgoing, inputs.io.deq.valid, fragments.io.enq.ready && pending +& outputs.io.count < 4.U)
  io.local.request.bits.address := Cat(offset(63, 4), 0.U(4.W))
  io.local.request.bits.write   := outgoing
  io.local.request.bits.data    := Mux(offset(3), Cat(word(63, 0), 0.U(64.W)), word)
  io.local.request.bits.mask    := Mux(half, Mux(offset(3), "hff00".U, "h00ff".U), "hffff".U)
  inputs.io.deq.ready           := writing && io.local.request.ready && io.local.request.valid && finish
  fragments.io.enq.valid        := reading && io.local.request.fire
  fragments.io.enq.bits.upper   := upper
  fragments.io.enq.bits.half    := half
  fragments.io.enq.bits.high    := offset(3)
  fragments.io.enq.bits.last    := last
  when(io.local.request.fire) {
    offset    := offset + amount
    remaining := Mux(last, 0.U, remaining - amount)
    upper     := !finish
  }
  val fragment = fragments.io.deq.bits
  val complete = fragment.upper || !fragment.half || fragment.last
  val piece    = Mux(fragment.high, io.local.response.bits.data(127, 64), io.local.response.bits.data(63, 0))
  io.local.response.ready        := state === active &&
    (outgoing || fragments.io.deq.valid && (!complete || outputs.io.enq.ready))
  fragments.io.deq.ready         := reading && io.local.response.fire
  outputs.io.enq.valid           := reading && io.local.response.valid && fragments.io.deq.valid && complete
  outputs.io.enq.bits            := Mux(
    fragment.half,
    Mux(fragment.upper, Cat(piece, low), Cat(0.U(64.W), piece)),
    io.local.response.bits.data
  )
  when(io.local.response.fire) {
    assert(pending > 0.U && !io.local.response.bits.error, "Payload storage access failed")
    when(reading && !complete)(low := piece)
  }
  pending                        := pending + io.local.request.fire - io.local.response.fire
  when(state === active && remaining === 0.U && pending === 0.U &&
    Mux(outgoing, inputs.io.count === 0.U, outputs.io.count === 0.U)) {
    state := reply
  }
  when(io.host.reply.fire)(state := idle)
}
