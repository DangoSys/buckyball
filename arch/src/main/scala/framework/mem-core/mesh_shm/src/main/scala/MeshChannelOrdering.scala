package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class MeshChannelOrdering(p: MeshSharedMemParams) extends Module {

  @public
  val io = IO(new Bundle {
    val client  = new MeshChannel(p)
    val ordered = Flipped(new MeshChannel(p))
  })

  val valid = RegInit(VecInit(Seq.fill(p.maxInFlight)(false.B)))
  val bank  = Reg(Vec(p.maxInFlight, UInt(p.bankBits.W)))
  val addr  = Reg(Vec(p.maxInFlight, UInt(p.addressBits.W)))
  val write = Reg(Vec(p.maxInFlight, Bool()))
  val tag   = Reg(Vec(p.maxInFlight, UInt(p.tagBits.W)))

  val request = io.client.request
  val free    = VecInit(valid.map(slot => !slot)).asUInt

  val conflict = VecInit((0 until p.maxInFlight).map { slot =>
    valid(slot) && bank(slot) === request.bits.bank && addr(slot) === request.bits.addr &&
    (write(slot) || request.bits.write)
  }).asUInt.orR

  val canIssue = free.orR && !conflict
  val freeSlot = PriorityEncoder(free)

  io.ordered.request.valid := request.valid && canIssue
  io.ordered.request.bits  := request.bits
  request.ready            := io.ordered.request.ready && canIssue
  io.client.response <> io.ordered.response

  val completed = VecInit((0 until p.maxInFlight).map(slot => valid(slot) && tag(slot) === io.client.response.bits.tag))
  when(io.client.response.fire) {
    assert(PopCount(completed) === 1.U, "Mesh response has no unique ordering entry")
  }

  for (slot <- 0 until p.maxInFlight) {
    when(io.client.response.fire && completed(slot)) {
      valid(slot) := false.B
    }
    when(request.fire && freeSlot === slot.U) {
      valid(slot) := true.B
      bank(slot)  := request.bits.bank
      addr(slot)  := request.bits.addr
      write(slot) := request.bits.write
      tag(slot)   := request.bits.tag
    }
  }
}
