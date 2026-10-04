package memcore.memory.coherence

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import memcore.bus.chi.RequestFlit
import memcore.memory.coherence.configs.CoherenceParams

class MshrAllocation(p: CoherenceParams) extends Bundle {
  val request = new RequestFlit(p.chi)
  // The controller supplies the resource key whose lifetime this transaction owns.
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
}

@instantiable
class Mshr(p: CoherenceParams) extends Module {
  @public val io = IO(new MshrIO(p))

  val valid      = RegInit(VecInit(Seq.fill(p.mshrEntries)(false.B)))
  val callerLive = RegInit(VecInit(Seq.fill(p.mshrEntries)(false.B)))
  val requests   = Reg(Vec(p.mshrEntries, new RequestFlit(p.chi)))
  val keys       = Reg(Vec(p.mshrEntries, UInt(p.lineAddressBits.W)))
  val free       = VecInit(valid.map(v => !v)).asUInt

  val conflicts = VecInit((0 until p.mshrEntries).map(i =>
    valid(i) && (keys(i) === io.allocate.bits.key ||
      (callerLive(i) && requests(i).srcId === io.allocate.bits.request.srcId &&
        requests(i).txnId === io.allocate.bits.request.txnId))
  ))

  val selected = if (p.mshrEntries == 1) 0.U else PriorityEncoder(free)

  io.allocate.ready := free.orR && !conflicts.asUInt.orR
  io.allocatedId    := selected
  io.outstanding    := PopCount(valid)
  for (i <- 0 until p.mshrEntries) {
    io.entries(i).valid   := valid(i)
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
    assert(valid(io.release.bits), "MSHR release targets a free slot")
    valid(io.release.bits)      := false.B
    callerLive(io.release.bits) := false.B
  }
  when(io.allocate.fire) {
    valid(selected)      := true.B
    callerLive(selected) := true.B
    requests(selected)   := io.allocate.bits.request
    keys(selected)       := io.allocate.bits.key
  }
}
