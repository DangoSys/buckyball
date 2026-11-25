package framework.generic.memory.coherence

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.bus.chi.RequestFlit
import framework.generic.memory.coherence.configs.CoherenceParams

class MshrAllocation(p: CoherenceParams) extends Bundle {
  val request = new RequestFlit(p.chi)
  // Earlier transactions with the controller's conflict key execute first.
  val key     = UInt(p.lineAddressBits.W)
}

class MshrEntry(p: CoherenceParams) extends Bundle {
  val valid   = Bool()
  val request = new RequestFlit(p.chi)
  val key     = UInt(p.lineAddressBits.W)
}

class MshrIO(p: CoherenceParams) extends Bundle {
  val allocate      = Flipped(Decoupled(new MshrAllocation(p)))
  val allocatedId   = Output(UInt(p.slotBits.W))
  val release       = Flipped(Valid(UInt(p.slotBits.W)))
  val releaseCaller = Input(UInt(p.mshrEntries.W))
  val entries       = Output(Vec(p.mshrEntries, new MshrEntry(p)))
  val outstanding   = Output(UInt(log2Ceil(p.mshrEntries + 1).W))
  // The address of every live transaction, waiting or not.
  val live          = Output(Vec(p.mshrEntries, Valid(UInt(p.chi.addressBits.W))))
}

@instantiable
class Mshr(p: CoherenceParams) extends Module {
  @public
  val io = IO(new MshrIO(p))

  val valid        = RegInit(VecInit(Seq.fill(p.mshrEntries)(false.B)))
  val predecessors = RegInit(VecInit(Seq.fill(p.mshrEntries)(0.U(p.mshrEntries.W))))
  val callerLive   = RegInit(VecInit(Seq.fill(p.mshrEntries)(false.B)))
  val requests     = Reg(Vec(p.mshrEntries, new RequestFlit(p.chi)))
  val keys         = Reg(Vec(p.mshrEntries, UInt(p.lineAddressBits.W)))
  val free         = VecInit(valid.map(v => !v)).asUInt

  val conflicts = VecInit((0 until p.mshrEntries).map(i =>
    valid(i) && callerLive(i) && requests(i).srcId === io.allocate.bits.request.srcId &&
      requests(i).txnId === io.allocate.bits.request.txnId
  ))

  val selected = if (p.mshrEntries == 1) 0.U else PriorityEncoder(free)

  io.allocate.ready := free.orR && !conflicts.asUInt.orR
  io.allocatedId    := selected
  io.outstanding    := PopCount(valid)
  for (i <- 0 until p.mshrEntries) {
    io.live(i).valid := valid(i)
    io.live(i).bits  := requests(i).addr
  }
  for (i <- 0 until p.mshrEntries) {
    io.entries(i).valid   := valid(i) && !predecessors(i).orR
    io.entries(i).request := requests(i)
    io.entries(i).key     := keys(i)
  }
  for (i <- 0 until p.mshrEntries) {
    when(io.releaseCaller(i)) {
      assert(valid(i) && callerLive(i), "MSHR caller release has no live caller")
      callerLive(i) := false.B
    }
  }
  when(io.release.valid) {
    assert(io.release.bits < p.mshrEntries.U, "MSHR release has unknown slot")
    assert(valid(io.release.bits) && !predecessors(io.release.bits).orR, "MSHR release targets an inactive slot")
    valid(io.release.bits)      := false.B
    callerLive(io.release.bits) := false.B
    for (i <- 0 until p.mshrEntries)
      predecessors(i) := predecessors(i) & ~UIntToOH(io.release.bits, p.mshrEntries)
  }
  when(io.allocate.fire) {
    valid(selected) := true.B
    val occupied = VecInit((0 until p.mshrEntries).map(i =>
      valid(i) && keys(i) === io.allocate.bits.key && !(io.release.valid && io.release.bits === i.U)
    )).asUInt
    predecessors(selected) := occupied
    callerLive(selected)   := true.B
    requests(selected)     := io.allocate.bits.request
    keys(selected)         := io.allocate.bits.key
  }
}
