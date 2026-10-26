package framework.ip.memory.mesh_shm

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

  val priorityBits = math.max(1, log2Ceil(channels))
  val priority     = RegInit(0.U(priorityBits.W))
  val available    = Wire(Vec(channels, Bool()))

  def distance(channel: Int): UInt = {
    val index = channel.U(priorityBits.W)
    Mux(index >= priority, index - priority, index + channels.U - priority)
  }

  for (channel <- 0 until channels) {
    val request  = io.client(channel).request
    val free     = VecInit(valid(channel).map(slot => !slot)).asUInt
    val conflict = VecInit(for {
      owner <- 0 until channels
      slot  <- 0 until p.maxInFlight
    } yield {
      valid(owner)(slot) && bank(owner)(slot) === request.bits.bank && addr(owner)(slot) === request.bits.addr &&
      (write(owner)(slot) || request.bits.write)
    }).asUInt.orR
    available(channel) := free.orR && !conflict
  }

  for (channel <- 0 until channels) {
    val request                = io.client(channel).request
    val free                   = VecInit(valid(channel).map(slot => !slot)).asUInt
    val higherPriorityConflict = (0 until channels).filter(_ != channel).map { owner =>
      val contender = io.client(owner).request
      contender.valid && available(owner) && contender.bits.bank === request.bits.bank &&
      contender.bits.addr === request.bits.addr && (contender.bits.write || request.bits.write) &&
      distance(owner) < distance(channel)
    }.reduceOption(_ || _).getOrElse(false.B)
    val canIssue               = available(channel) && !higherPriorityConflict
    val freeSlot               = PriorityEncoder(free)

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

  when(VecInit(io.client.map(_.request.fire)).asUInt.orR) {
    priority := Mux(priority === (channels - 1).U, 0.U, priority + 1.U)
  }
}
