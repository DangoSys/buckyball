package memcore.memory.coherence

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import memcore.memory.coherence.configs.CoherenceParams

class DirectoryEntry(p: CoherenceParams) extends Bundle {
  val sharers = UInt(p.agents.W)
  val unique  = Bool()
}

class DirectoryAddress(p: CoherenceParams) extends Bundle {
  val set = UInt(p.cache.setBits.W)
  val way = UInt(p.cache.wayBits.W)
}

class DirectoryUpdate(p: CoherenceParams) extends Bundle {
  val address = new DirectoryAddress(p)
  val entry   = new DirectoryEntry(p)
}

class DirectoryIO(p: CoherenceParams) extends Bundle {
  val read    = Input(Vec(p.mshrEntries, new DirectoryAddress(p)))
  val entries = Output(Vec(p.mshrEntries, new DirectoryEntry(p)))
  val update  = Flipped(Vec(p.mshrEntries, Valid(new DirectoryUpdate(p))))
}

@instantiable
class Directory(p: CoherenceParams) extends Module {
  @public
  val io      = IO(new DirectoryIO(p))
  val entries =
    RegInit(VecInit(Seq.fill(p.cache.sets)(VecInit(Seq.fill(p.cache.ways)(0.U.asTypeOf(new DirectoryEntry(p)))))))
  for (i <- 0 until p.mshrEntries) {
    io.entries(i) := entries(io.read(i).set)(io.read(i).way)
  }
  for {
    set  <- 0 until p.cache.sets
    way  <- 0 until p.cache.ways
  } {
    val writers = VecInit(io.update.map(u => u.valid && u.bits.address.set === set.U && u.bits.address.way === way.U))
    assert(PopCount(writers) <= 1.U, "Multiple transactions update one directory entry")
    when(writers.asUInt.orR) {
      val next = Mux1H(writers, io.update.map(_.bits.entry))
      assert(!next.unique || PopCount(next.sharers) === 1.U, "Unique ownership requires exactly one CPU")
      entries(set)(way) := next
    }
  }
}
