package framework.generic.memory.coherence

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.memory.coherence.configs.CoherenceParams

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
  val read   = Input(new DirectoryAddress(p))
  val entry  = Output(new DirectoryEntry(p))
  val update = Flipped(Vec(p.cache.sets, Vec(p.cache.ways, Valid(new DirectoryEntry(p)))))
  val probe  = Input(new DirectoryAddress(p))
  val probed = Output(new DirectoryEntry(p))
}

@instantiable
class Directory(p: CoherenceParams) extends Module {
  @public
  val io      = IO(new DirectoryIO(p))
  val entries =
    RegInit(VecInit(Seq.fill(p.cache.sets)(VecInit(Seq.fill(p.cache.ways)(0.U.asTypeOf(new DirectoryEntry(p)))))))
  io.probed := entries(io.probe.set)(io.probe.way)
  io.entry  := entries(io.read.set)(io.read.way)
  for {
    set <- 0 until p.cache.sets
    way <- 0 until p.cache.ways
  } {
    val update = io.update(set)(way)
    when(update.valid) {
      assert(!update.bits.unique || PopCount(update.bits.sharers) === 1.U, "Unique ownership requires exactly one CPU")
      entries(set)(way) := update.bits
    }
  }
}
