package framework.generic.bus.axi4

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/**
 * A master of whole `narrow` beats on a port of `wide` beats, `rows` of them a beat: a burst of
 * up to `beats` narrow beats, or one narrow beat of any size. A read burst is fetched in wide
 * beats, which leave the wide port free after a `rows`-th of the narrow beats' time, and is
 * returned a narrow beat a cycle as its wide beats arrive. A write burst is collected whole,
 * then written in wide beats whose strobes keep the bytes outside it. A wide beat of one row is
 * the narrow port itself.
 *
 * The bursts wait in single-port memories. A write burst is one: written while it is collected,
 * read a beat ahead while it is sent. A read burst is two, its even and its odd wide beats: a
 * beat arrives a cycle at most, so each is free every other cycle for the read of the beat the
 * narrow side reaches next, which is back before the `rows` narrow beats of the one before it
 * are out. The beat the narrow side is in and the one after it are registers, and a beat that
 * arrives for either goes straight there.
 */
@instantiable
class Widen(narrow: Params, wide: Params, beats: Int) extends Module {
  require(narrow.addressBits == wide.addressBits && narrow.idBits == wide.idBits)
  require(wide.dataBits % narrow.dataBits == 0)
  val rows      = wide.dataBits / narrow.dataBits
  val depth     = beats / rows + 1
  val laneBits  = log2Ceil(rows)
  val rowBits   = log2Ceil(narrow.bytes)
  val half      = (depth + 1) / 2
  // A memory of one beat still has an address bit.
  val depthBits = log2Ceil(depth).max(1)
  val halfBits  = log2Ceil(half).max(1)
  require(rows == 1 || rows >= 4 || beats == 1, "A wide beat lasts the cycles the read of the next one takes")

  @public
  val io = IO(new Bundle {
    val in  = Flipped(new Port(narrow))
    val out = new Port(wide)
  })

  if (rows == 1) io.out <> io.in
  else {
    def lane(address: UInt): UInt = address(rowBits + laneBits - 1, rowBits)
    def base(address: UInt): UInt = Cat(address(wide.addressBits - 1, rowBits + laneBits), 0.U((rowBits + laneBits).W))

    // Read: the wide beats into the memories, the narrow beats out of `current` in order.
    val rIdle :: rIssue :: rData :: Nil = Enum(3)
    val rState                          = RegInit(rIdle)
    val read                            = Reg(new Address(narrow))
    val fetched                         = Reg(UInt(log2Ceil(depth + 1).W))
    val returned                        = Reg(UInt(9.W))
    val readError                       = Reg(UInt(2.W))
    val evenBeats                       = SyncReadMem(half, UInt(wide.dataBits.W))
    val oddBeats                        = SyncReadMem(half, UInt(wide.dataBits.W))
    val current                         = Reg(UInt(wide.dataBits.W))
    val currentValid                    = RegInit(false.B)
    val following                       = Reg(UInt(wide.dataBits.W))
    val followingValid                  = RegInit(false.B)
    val recalling                       = RegInit(false.B)
    val recalled                        = Reg(Bool())
    val readLast                        = (lane(read.addr) +& read.len) >> laneBits

    io.in.ar.ready              := rState === rIdle
    when(io.in.ar.fire) {
      assert(io.in.ar.bits.burst === 1.U && io.in.ar.bits.len < beats.U &&
        (io.in.ar.bits.size === rowBits.U || io.in.ar.bits.len === 0.U))
      read           := io.in.ar.bits; fetched := 0.U; returned := 0.U; readError := 0.U; rState := rIssue
      currentValid   := false.B
      followingValid := false.B
    }
    io.out.ar.valid             := rState === rIssue
    io.out.ar.bits              := read
    io.out.ar.bits.addr         := base(read.addr)
    io.out.ar.bits.len          := readLast
    io.out.ar.bits.size         := log2Ceil(wide.bytes).U
    when(io.out.ar.fire)(rState := rData)
    io.out.r.ready              := rState === rData && fetched <= readLast

    // The wide beat the narrow side is in, and the one it is in next cycle.
    val position    = lane(read.addr) +& returned
    val index       = position >> laneBits
    val leaving     = io.in.r.fire && position(laneBits - 1, 0) === (rows - 1).U
    val reached     = index +& leaving
    val wanted      = index +& 1.U
    // An arriving beat goes to the register that waits for it, or to its memory.
    val toCurrent   = io.out.r.fire && fetched === reached && (leaving || !currentValid)
    val toFollowing = io.out.r.fire && fetched === reached +& 1.U
    val store       = io.out.r.fire && !toCurrent && !toFollowing
    // The beat after `current` comes back from its memory in a cycle the memory takes no arriving beat.
    val recall      = rState === rData && !followingValid && !recalling && !leaving && wanted < fetched &&
      !(store && fetched(0) === wanted(0))
    val stored      = Seq(evenBeats, oddBeats).zipWithIndex.map { case (memory, parity) =>
      val write = store && fetched(0) === parity.U
      memory.readWrite(
        Mux(write, fetched >> 1, wanted >> 1)(halfBits - 1, 0),
        io.out.r.bits.data,
        write || recall && wanted(0) === parity.U,
        write
      )
    }
    when(io.out.r.fire) {
      readError := readError | io.out.r.bits.resp
      fetched   := fetched + 1.U
    }
    when(leaving) {
      assert(!recalling, "A wide beat is back before the one before it is out")
      current        := following
      currentValid   := followingValid
      followingValid := false.B
    }
    when(recall) {
      recalling := true.B
      recalled  := wanted(0)
    }
    when(recalling) {
      recalling      := false.B
      following      := Mux(recalled, stored(1), stored(0))
      followingValid := true.B
    }
    when(toCurrent) {
      current      := io.out.r.bits.data
      currentValid := true.B
    }
    when(toFollowing) {
      following      := io.out.r.bits.data
      followingValid := true.B
    }
    when(rState === rData)(assert(
      currentValid === (index < fetched),
      "A narrow beat is out the cycle after its wide beat"
    ))
    io.in.r.valid := rState === rData && currentValid
    io.in.r.bits.id   := read.id
    io.in.r.bits.data := (current >> (position(laneBits - 1, 0) * narrow.dataBits.U))(narrow.dataBits - 1, 0)
    io.in.r.bits.resp := readError
    io.in.r.bits.last := returned === read.len
    when(io.in.r.fire) {
      returned                       := returned + 1.U
      when(io.in.r.bits.last)(rState := rIdle)
    }

    // Write: the narrow beats packed into the memory, then the wide beats.
    val wIdle :: wCollect :: wIssue :: wSend :: wResponse :: wReturn :: Nil = Enum(6)
    val wState                                                              = RegInit(wIdle)
    val write                                                               = Reg(new Address(narrow))
    val collected                                                           = Reg(UInt(9.W))
    val sent                                                                = Reg(UInt(log2Ceil(depth + 1).W))
    val packData                                                            = Reg(Vec(rows, UInt(narrow.dataBits.W)))
    val packStrobe                                                          = Reg(Vec(rows, UInt(narrow.bytes.W)))
    val writeBeats                                                          = SyncReadMem(depth, UInt((wide.dataBits + wide.bytes).W))
    val writeResponse                                                       = Reg(UInt(2.W))
    val writeLast                                                           = (lane(write.addr) +& write.len) >> laneBits

    io.in.aw.ready := wState === wIdle
    when(io.in.aw.fire) {
      assert(io.in.aw.bits.burst === 1.U && io.in.aw.bits.len < beats.U &&
        (io.in.aw.bits.size === rowBits.U || io.in.aw.bits.len === 0.U))
      write                := io.in.aw.bits; collected := 0.U; sent := 0.U; wState := wCollect
      packStrobe.foreach(_ := 0.U)
    }
    val place = lane(write.addr) +& collected
    val data   = WireDefault(packData)
    val strobe = WireDefault(packStrobe)
    data(place(laneBits - 1, 0))   := io.in.w.bits.data
    strobe(place(laneBits - 1, 0)) := io.in.w.bits.strb
    val keep = io.in.w.fire && (place(laneBits - 1, 0) === (rows - 1).U || io.in.w.bits.last)
    io.in.w.ready := wState === wCollect
    when(io.in.w.fire) {
      assert(io.in.w.bits.last === (collected === write.len), "Burst length mismatch")
      packData                       := data
      packStrobe                     := strobe
      when(keep)(packStrobe.foreach(_ := 0.U))
      collected                      := collected + 1.U
      when(io.in.w.bits.last)(wState := wIssue)
    }
    // The wide beat the port shows next cycle is read while the address and the beat before it go.
    val finishing = io.out.w.fire && io.out.w.bits.last
    val beat = writeBeats.readWrite(
      Mux(keep, place >> laneBits, sent + io.out.w.fire)(depthBits - 1, 0),
      Cat(strobe.asUInt, data.asUInt),
      keep || (wState === wIssue || wState === wSend) && !finishing,
      keep
    )
    io.out.aw.valid             := wState === wIssue
    io.out.aw.bits              := write
    io.out.aw.bits.addr         := base(write.addr)
    io.out.aw.bits.len          := writeLast
    io.out.aw.bits.size         := log2Ceil(wide.bytes).U
    when(io.out.aw.fire)(wState := wSend)
    io.out.w.valid              := wState === wSend
    io.out.w.bits.data          := beat(wide.dataBits - 1, 0)
    io.out.w.bits.strb          := beat >> wide.dataBits
    io.out.w.bits.last          := sent === writeLast
    when(io.out.w.fire) {
      sent                            := sent + 1.U
      when(io.out.w.bits.last)(wState := wResponse)
    }
    io.out.b.ready              := wState === wResponse
    when(io.out.b.fire) { writeResponse := io.out.b.bits.resp; wState := wReturn }
    io.in.b.valid               := wState === wReturn
    io.in.b.bits.id             := write.id
    io.in.b.bits.resp           := writeResponse
    when(io.in.b.fire)(wState   := wIdle)
  }
}
