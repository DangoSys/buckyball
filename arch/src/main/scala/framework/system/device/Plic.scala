package framework.system.device

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import chisel3.util._

case class PlicParams(
  base:         BigInt = BigInt("c000000", 16),
  bytes:        BigInt = BigInt("4000000", 16),
  sources:      Int = 1,
  priorityBits: Int = 3) {
  require(sources >= 1 && sources <= 31 && priorityBits >= 1 && priorityBits <= 32)
}

/**
 * RISC-V PLIC with level-triggered gateways and 32-bit registers. Hart h owns context 2h (M) and
 * 2h+1 (S). A claim read clears the source's pending bit and holds it in service until completed.
 */
@instantiable
class Plic(p: PlicParams, hartIds: Seq[Int]) extends Module {
  require(hartIds.distinct.size == hartIds.size)
  val n        = hartIds.size
  val contexts = 2 * (hartIds.max + 1)

  @public val io = IO(new Bundle {
    val port    = new DevicePort
    val sources = Input(UInt(p.sources.W))
    val meip    = Output(Vec(n, Bool()))
    val seip    = Output(Vec(n, Bool()))
  })

  // Index 0 is the reserved source zero throughout.
  val priority  = RegInit(VecInit(Seq.fill(p.sources + 1)(0.U(p.priorityBits.W))))
  val pending   = RegInit(VecInit(Seq.fill(p.sources + 1)(false.B)))
  val inService = RegInit(VecInit(Seq.fill(p.sources + 1)(false.B)))
  val enable    = RegInit(VecInit(Seq.fill(contexts)(0.U((p.sources + 1).W))))
  val threshold = RegInit(VecInit(Seq.fill(contexts)(0.U(p.priorityBits.W))))

  for (s <- 1 to p.sources) {
    when(io.sources(s - 1) && !inService(s))(pending(s) := true.B)
  }

  /** Highest-priority enabled pending source above the context threshold; ties take the lowest id. */
  def best(context: Int): (Bool, UInt) = {
    val candidates = (1 to p.sources).map { s =>
      (pending(s) && enable(context)(s) && priority(s) > threshold(context), priority(s), s.U(5.W))
    }
    candidates.foldLeft((false.B, 0.U(p.priorityBits.W), 0.U(5.W))) { case ((v, pr, id), (cv, cpr, cid)) =>
      val take = cv && (!v || cpr > pr)
      (v || cv, Mux(take, cpr, pr), Mux(take, cid, id))
    } match { case (v, _, id) => (v, id) }
  }

  val claims = (0 until contexts).map(best)

  for ((hart, i) <- hartIds.zipWithIndex) {
    io.meip(i) := claims(2 * hart)._1
    io.seip(i) := claims(2 * hart + 1)._1
  }

  val access = io.port.access.bits
  val offset = access.addr - p.base.U
  val valid  = io.port.access.valid && access.size === 2.U && offset(1, 0) === 0.U
  val data   = access.data(31, 0)

  val hits  = Wire(Vec(p.sources + 1 + 3 * contexts, Bool()))
  val reads = Wire(Vec(p.sources + 1 + 3 * contexts, UInt(32.W)))

  def at(slot: Int, address: BigInt, value: UInt): Bool = {
    hits(slot)  := offset === address.U
    reads(slot) := value.pad(32)
    valid && hits(slot)
  }

  for (s <- 1 to p.sources) {
    when(at(s, 4 * s, priority(s)) && access.write)(priority(s) := data)
  }
  at(0, 0x1000, pending.asUInt)
  for (c <- 0 until contexts) {
    val slot = p.sources + 1 + 3 * c
    when(at(slot, 0x2000 + 0x80 * c, enable(c)) && access.write)(enable(c)                       := data(p.sources, 1) ## 0.U(1.W))
    when(at(slot + 1, 0x200000 + 0x1000 * BigInt(c), threshold(c)) && access.write)(threshold(c) := data)
    when(at(slot + 2, 0x200004 + 0x1000 * BigInt(c), claims(c)._2)) {
      when(access.write) {
        // Completion releases the gateway; an id that is not enabled for this context is ignored.
        for (s <- 1 to p.sources) {
          when(data === s.U && enable(c)(s))(inService(s) := false.B)
        }
      }.elsewhen(claims(c)._1) {
        for (s <- 1 to p.sources) {
          when(claims(c)._2 === s.U) { pending(s) := false.B; inService(s) := true.B }
        }
      }
    }
  }

  io.port.read  := Mux1H(hits, reads)
  io.port.error := !(valid && hits.asUInt.orR)
}
