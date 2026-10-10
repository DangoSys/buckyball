package memcore.bus.axi4

import memcore.memory.queue.Queue

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

class AddressAcceptance(p: Params, masters: Int) extends Bundle {
  val master   = UInt(math.max(1, log2Ceil(masters)).W)
  val original = new Address(p)
  val slot     = UInt(p.idBits.W)
}

/** Registers an arbitrated address before publishing it to the DDR port. */
@instantiable
class AddressSelector(p: Params, masters: Int) extends Module {
  require(masters > 0)
  val masterBits = math.max(1, log2Ceil(masters))

  @public val io = IO(new Bundle {
    val in       = Vec(masters, Flipped(Decoupled(new Address(p))))
    val enabled  = Input(Vec(masters, Bool()))
    val slot     = Input(UInt(p.idBits.W))
    val out      = Decoupled(new Address(p))
    val accepted = Output(Valid(new AddressAcceptance(p, masters)))
  })

  val cursor   = RegInit(0.U(masterBits.W))
  val held     = RegInit(false.B)
  val address  = Reg(new Address(p))
  val eligible = VecInit(io.in.indices.map(i => io.in(i).valid && io.enabled(i)))
  val ordered  = VecInit(io.in.indices.map(offset => eligible((cursor +& offset.U) % masters.U)))
  val offset   = if (masters == 1) 0.U(masterBits.W) else PriorityEncoder(ordered)
  val selected = (cursor +& offset) % masters.U
  val capture  = (!held || io.out.ready) && eligible.asUInt.orR

  io.out.valid := held
  io.out.bits  := address
  for (i <- io.in.indices) {
    io.in(i).ready := capture && selected === i.U
  }
  io.accepted.valid := capture
  io.accepted.bits.master   := selected
  io.accepted.bits.original := VecInit(io.in.map(_.bits))(selected)
  io.accepted.bits.slot     := io.slot

  when(io.out.fire)(held := false.B)
  when(capture) {
    address    := io.accepted.bits.original
    address.id := io.slot
    held       := true.B
    cursor     := Mux(selected === (masters - 1).U, 0.U, selected + 1.U)
  }
}

class WriteOwner(p: Params, masters: Int) extends Bundle {
  val master = UInt(math.max(1, log2Ceil(masters)).W)
  val slot   = UInt(p.idBits.W)
}

/** Normal AXI4 transfers, dynamically mapped onto a bounded DDR ID space. */
@instantiable
class Interconnect(p: Params, masters: Int, slots: Option[Int] = None) extends Module {
  require(masters > 0)

  val capacity = slots.getOrElse {
    val ids = BigInt(1) << p.idBits
    require(ids.isValidInt, "A full AXI ID pool must fit the slot index; supply an explicit slot count")
    ids.toInt
  }

  require(capacity > 1 && BigInt(capacity) <= (BigInt(1) << p.idBits))
  val masterBits = math.max(1, log2Ceil(masters))

  @public val io = IO(new Bundle {
    val in          = Vec(masters, Flipped(new Port(p)))
    val out         = new Port(p)
    val outstanding = Output(UInt(log2Ceil(2 * capacity + 1).W))
  })

  val readLive       = RegInit(VecInit(Seq.fill(capacity)(false.B)))
  val readPosted     = RegInit(VecInit(Seq.fill(capacity)(false.B)))
  val readMaster     = Reg(Vec(capacity, UInt(masterBits.W)))
  val readId         = Reg(Vec(capacity, UInt(p.idBits.W)))
  val readRemaining  = Reg(Vec(capacity, UInt(9.W)))
  val writeLive      = RegInit(VecInit(Seq.fill(capacity)(false.B)))
  val writePosted    = RegInit(VecInit(Seq.fill(capacity)(false.B)))
  val writeDone      = RegInit(VecInit(Seq.fill(capacity)(false.B)))
  val writeMaster    = Reg(Vec(capacity, UInt(masterBits.W)))
  val writeId        = Reg(Vec(capacity, UInt(p.idBits.W)))
  val writeRemaining = Reg(Vec(capacity, UInt(9.W)))

  io.outstanding := PopCount(readLive) +& PopCount(writeLive)

  val reads:  Instance[AddressSelector] = Instantiate(new AddressSelector(p, masters))
  val writes: Instance[AddressSelector] = Instantiate(new AddressSelector(p, masters))
  val owners    = Module(new Queue(new WriteOwner(p, masters), capacity))
  val readFree  = VecInit(readLive.map(live => !live))
  val writeFree = VecInit(writeLive.map(live => !live))
  reads.io.slot  := PriorityEncoder(readFree)
  writes.io.slot := PriorityEncoder(writeFree)
  io.out.ar <> reads.io.out
  io.out.aw <> writes.io.out

  for (master <- 0 until masters) {
    reads.io.in(master) <> io.in(master).ar
    writes.io.in(master) <> io.in(master).aw
    // Different DDR IDs may complete out of order. Keep one transaction per
    // original master/ID live so restoration preserves AXI's same-ID ordering.
    val readBusy  = readLive.indices.map(slot =>
      readLive(slot) && readMaster(slot) === master.U && readId(slot) === io.in(master).ar.bits.id
    ).reduce(_ || _)
    val writeBusy = writeLive.indices.map(slot =>
      writeLive(slot) && writeMaster(slot) === master.U && writeId(slot) === io.in(master).aw.bits.id
    ).reduce(_ || _)
    reads.io.enabled(master)  := readFree.asUInt.orR && !readBusy
    writes.io.enabled(master) := writeFree.asUInt.orR && owners.io.enq.ready && !writeBusy
    when(io.in(master).ar.valid) {
      assert(!io.in(master).ar.bits.lock, "AXI exclusive read is outside the dynamically mapped interconnect contract")
    }
    when(io.in(master).aw.valid) {
      assert(!io.in(master).aw.bits.lock, "AXI exclusive write is outside the dynamically mapped interconnect contract")
    }
  }

  when(reads.io.accepted.valid) {
    val accepted = reads.io.accepted.bits
    readLive(accepted.slot)      := true.B
    readPosted(accepted.slot)    := false.B
    readMaster(accepted.slot)    := accepted.master
    readId(accepted.slot)        := accepted.original.id
    readRemaining(accepted.slot) := accepted.original.len +& 1.U
  }
  when(writes.io.accepted.valid) {
    val accepted = writes.io.accepted.bits
    writeLive(accepted.slot)      := true.B
    writePosted(accepted.slot)    := false.B
    writeDone(accepted.slot)      := false.B
    writeMaster(accepted.slot)    := accepted.master
    writeId(accepted.slot)        := accepted.original.id
    writeRemaining(accepted.slot) := accepted.original.len +& 1.U
  }
  when(io.out.ar.fire)(readPosted(io.out.ar.bits.id)  := true.B)
  when(io.out.aw.fire)(writePosted(io.out.aw.bits.id) := true.B)

  owners.io.enq.valid       := writes.io.accepted.valid
  owners.io.enq.bits.master := writes.io.accepted.bits.master
  owners.io.enq.bits.slot   := writes.io.accepted.bits.slot
  val head = owners.io.deq.bits
  // WVALID never waits for DDR AWREADY: a slave can buffer W before AW.
  // Accepted AW order determines W ownership even while an address is stalled.
  io.out.w.valid := owners.io.deq.valid && VecInit(io.in.map(_.w.valid))(head.master)
  io.out.w.bits  := VecInit(io.in.map(_.w.bits))(head.master)
  for (master <- 0 until masters) {
    io.in(master).w.ready := owners.io.deq.valid && head.master === master.U && io.out.w.ready
  }
  owners.io.deq.ready := io.out.w.fire && io.out.w.bits.last
  when(io.out.w.fire) {
    assert(writeLive(head.slot) && !writeDone(head.slot), "AXI write data has no live address owner")
    assert(io.out.w.bits.last === (writeRemaining(head.slot) === 1.U), "AXI WLAST does not match AWLEN")
    when(io.out.w.bits.last) {
      writeDone(head.slot) := true.B
    }.otherwise {
      writeRemaining(head.slot) := writeRemaining(head.slot) - 1.U
    }
  }

  val rSlot   = io.out.r.bits.id
  val rKnown  = rSlot < capacity.U && readLive(rSlot)
  val rPosted = readPosted(rSlot) || (io.out.ar.fire && io.out.ar.bits.id === rSlot)
  io.out.r.ready := rKnown && rPosted && VecInit(io.in.map(_.r.ready))(readMaster(rSlot))
  for (master <- 0 until masters) {
    io.in(master).r.valid   := io.out.r.valid && rKnown && rPosted && readMaster(rSlot) === master.U
    io.in(master).r.bits    := io.out.r.bits
    io.in(master).r.bits.id := readId(rSlot)
  }
  when(io.out.r.valid) {
    assert(rKnown && rPosted, "AXI read response refers to an unissued ID")
  }
  when(io.out.r.fire) {
    assert(io.out.r.bits.last === (readRemaining(rSlot) === 1.U), "AXI RLAST does not match ARLEN")
    when(io.out.r.bits.last) {
      readLive(rSlot)   := false.B
      readPosted(rSlot) := false.B
    }.otherwise {
      readRemaining(rSlot) := readRemaining(rSlot) - 1.U
    }
  }

  val bSlot   = io.out.b.bits.id
  val bKnown  = bSlot < capacity.U && writeLive(bSlot)
  val bPosted = writePosted(bSlot) || (io.out.aw.fire && io.out.aw.bits.id === bSlot)
  val bDone   = writeDone(bSlot) || (io.out.w.fire && io.out.w.bits.last && head.slot === bSlot)
  io.out.b.ready := bKnown && bPosted && bDone && VecInit(io.in.map(_.b.ready))(writeMaster(bSlot))
  for (master <- 0 until masters) {
    io.in(master).b.valid   := io.out.b.valid && bKnown && bPosted && bDone && writeMaster(bSlot) === master.U
    io.in(master).b.bits    := io.out.b.bits
    io.in(master).b.bits.id := writeId(bSlot)
  }
  when(io.out.b.valid) {
    assert(bKnown && bPosted && bDone, "AXI write response preceded its address or final data")
  }
  when(io.out.b.fire) {
    writeLive(bSlot)   := false.B
    writePosted(bSlot) := false.B
    writeDone(bSlot)   := false.B
  }
}
