package framework.system.tile.tlink

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import memcore.bus.axi4

/** Tile-addressed shared-storage fabric, separate from the DDR and CPU coherence paths. */
@instantiable
class Fabric(p: axi4.Params, tileIds: Seq[Int], sharedBytes: BigInt) extends Module {
  require(tileIds.nonEmpty && tileIds.distinct.size == tileIds.size && tileIds.forall(_ >= 0))
  require(p.dataBits == 128 && sharedBytes >= p.bytes)
  val localBits = log2Ceil(sharedBytes)
  val indexBits = math.max(1, log2Ceil(tileIds.size))
  require(p.addressBits >= localBits + log2Ceil(tileIds.max + 1))

  @public val io = IO(new Bundle {
    val source = Vec(tileIds.size, Flipped(new axi4.Port(p)))
    val target = Vec(tileIds.size, new axi4.Port(p))
  })

  val targets = tileIds.indices.map(_ => Instantiate(new axi4.Interconnect(p, tileIds.size)))
  for (i <- tileIds.indices) { io.target(i) <> targets(i).io.out }

  for (source <- tileIds.indices) {
    val port           = io.source(source)
    val reading        = RegInit(false.B)
    val writing        = RegInit(false.B)
    val readTarget     = Reg(UInt(indexBits.W))
    val writeTarget    = Reg(UInt(indexBits.W))
    val readMatches    = VecInit(tileIds.map(id => (port.ar.bits.addr >> localBits) === id.U))
    val writeMatches   = VecInit(tileIds.map(id => (port.aw.bits.addr >> localBits) === id.U))
    val readSelection  = PriorityEncoder(readMatches)
    val writeSelection = PriorityEncoder(writeMatches)

    def check(address: axi4.Address, matched: Bool): Unit = {
      val offset = address.addr(localBits - 1, 0)
      val bytes  = (address.len +& 1.U) << log2Ceil(p.bytes)
      assert(matched, "TLink address does not select a shared-storage tile")
      assert(
        address.size === log2Ceil(p.bytes).U && address.burst === 1.U && !address.lock,
        "TLink requires full-width INCR bursts"
      )
      assert(
        offset(log2Ceil(p.bytes) - 1, 0) === 0.U && (offset +& bytes) <= sharedBytes.U,
        "TLink burst exceeds shared storage or is unaligned"
      )
      assert((offset.pad(12)(11, 0) +& bytes) <= 4096.U, "TLink burst crosses a 4KiB boundary")
    }
    when(port.ar.valid && !reading)(check(port.ar.bits, readMatches.asUInt.orR))
    when(port.aw.valid && !writing)(check(port.aw.bits, writeMatches.asUInt.orR))
    when(port.ar.fire) { reading := true.B; readTarget := readSelection }
    when(port.r.fire && port.r.bits.last)(reading := false.B)
    when(port.aw.fire) { writing := true.B; writeTarget := writeSelection }
    when(port.b.fire)(writing                     := false.B)

    port.ar.ready := !reading && readMatches.asUInt.orR && VecInit(targets.map(_.io.in(source).ar.ready))(readSelection)
    port.aw.ready := !writing && writeMatches.asUInt.orR && VecInit(targets.map(_.io.in(source).aw.ready))(
      writeSelection
    )
    port.w.ready  := writing && VecInit(targets.map(_.io.in(source).w.ready))(writeTarget)
    port.r.valid  := reading && VecInit(targets.map(_.io.in(source).r.valid))(readTarget)
    port.r.bits   := VecInit(targets.map(_.io.in(source).r.bits))(readTarget)
    port.b.valid  := writing && VecInit(targets.map(_.io.in(source).b.valid))(writeTarget)
    port.b.bits   := VecInit(targets.map(_.io.in(source).b.bits))(writeTarget)
    for (target <- tileIds.indices) {
      val route = targets(target).io.in(source)
      route.ar.valid     := port.ar.valid && !reading && readMatches(target)
      route.ar.bits      := port.ar.bits
      route.ar.bits.addr := port.ar.bits.addr(localBits - 1, 0)
      route.aw.valid     := port.aw.valid && !writing && writeMatches(target)
      route.aw.bits      := port.aw.bits
      route.aw.bits.addr := port.aw.bits.addr(localBits - 1, 0)
      route.w.valid      := port.w.valid && writing && writeTarget === target.U
      route.w.bits       := port.w.bits
      route.r.ready      := port.r.ready && reading && readTarget === target.U
      route.b.ready      := port.b.ready && writing && writeTarget === target.U
    }
  }
}
