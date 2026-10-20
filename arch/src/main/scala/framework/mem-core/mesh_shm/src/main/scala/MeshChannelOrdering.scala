package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class MeshChannelOrdering(p: MeshSharedMemParams, channels: Int) extends Module {
  require(channels > 0)

  @public
  val io = IO(new Bundle {
    val client  = Vec(channels, new MeshChannel(p))
    val ordered = Vec(channels, Flipped(new MeshChannel(p)))
  })

  val valid = RegInit(VecInit(Seq.fill(channels)(VecInit(Seq.fill(p.maxInFlight)(false.B)))))
  val bank  = Reg(Vec(channels, Vec(p.maxInFlight, UInt(p.bankBits.W))))
  val addr  = Reg(Vec(channels, Vec(p.maxInFlight, UInt(p.addressBits.W))))
  val write = Reg(Vec(channels, Vec(p.maxInFlight, Bool())))
  val tag   = Reg(Vec(channels, Vec(p.maxInFlight, UInt(p.tagBits.W))))

  val reserved = Wire(Vec(channels, Bool()))

  for (channel <- 0 until channels) {
    val request = io.client(channel).request
    val free    = VecInit(valid(channel).map(slot => !slot)).asUInt

    val outstandingConflict = VecInit(for {
      owner <- 0 until channels
      slot  <- 0 until p.maxInFlight
    } yield {
      valid(owner)(slot) && bank(owner)(slot) === request.bits.bank && addr(owner)(slot) === request.bits.addr &&
      (write(owner)(slot) || request.bits.write)
    }).asUInt.orR
    val reservedConflict    = (0 until channel).map { owner =>
      val earlier = io.client(owner).request.bits
      reserved(owner) && earlier.bank === request.bits.bank && earlier.addr === request.bits.addr &&
      (earlier.write || request.bits.write)
    }.reduceOption(_ || _).getOrElse(false.B)
    val canIssue            = free.orR && !outstandingConflict && !reservedConflict
    val freeSlot            = PriorityEncoder(free)

    reserved(channel)                 := request.valid && canIssue
    io.ordered(channel).request.valid := request.valid && canIssue
    io.ordered(channel).request.bits  := request.bits
    request.ready                     := io.ordered(channel).request.ready && canIssue
    io.client(channel).response <> io.ordered(channel).response

    val completed = VecInit((0 until p.maxInFlight).map(slot =>
      valid(channel)(slot) && tag(channel)(slot) === io.client(channel).response.bits.tag
    ))
    when(io.client(channel).response.fire) {
      assert(PopCount(completed) === 1.U, "Mesh response has no unique ordering entry")
    }

    for (slot <- 0 until p.maxInFlight) {
      when(io.client(channel).response.fire && completed(slot)) {
        valid(channel)(slot) := false.B
      }
      when(request.fire && freeSlot === slot.U) {
        valid(channel)(slot) := true.B
        bank(channel)(slot)  := request.bits.bank
        addr(channel)(slot)  := request.bits.addr
        write(channel)(slot) := request.bits.write
        tag(channel)(slot)   := request.bits.tag
      }
    }
  }
}
