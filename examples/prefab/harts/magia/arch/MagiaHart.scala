package examples.hart.magia

import chisel3._
import chisel3.util._
import framework.generic.queue.Queue
import framework.generic.bus.axi4
import framework.generic.bus.chi.HomeMapping
import framework.generic.acc.debug.trace.MTraceDPI
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.hart.hlink.{HLinkIO, HasHLink}
import framework.hart.configs.HartParam
import framework.hart.cbus.HartFactory
import buckyball.config.{Design, HartPlacement}
import framework.generic.contract.sims.configs.SimParam
import framework.mlp.fetch.configs.{Params => FetchParam}

@instantiable
class MagiaHart(p: HartParam, fetch: FetchParam) extends Module with HasHLink {
  require(p.hlink.uncached == axi4.Params(44, 128, 6))
  @public
  val hlink = IO(new HLinkIO(p.hlink))
  val loom  = Instantiate(new Loom(p, fetch))
  loom.io.interrupts := hlink.interrupts(0)
  hlink.memory.coherent.take(2).zip(loom.io.coherent).foreach { case (sink, source) => sink <> source }
  val native = Instantiate(new MeshRTL)
  native.io.clock    := clock
  // Preserve the system reset edge for gated domains, and reset on job entry and exit.
  val runningChanged = loom.io.running =/= RegNext(loom.io.running, false.B)
  native.io.reset        := reset.asBool || runningChanged
  native.io.fetch_enable := loom.io.running
  native.io.boot_address := loom.io.boot
  val memory = Instantiate(new axi4.Decoder(Seq((BigInt("c0000000", 16), BigInt(0x40000000))), p.hlink.ddr))

  val dma = Instantiate(new DmaRequester(
    p.hlink.ddr,
    p.hlink.line,
    3,
    HomeMapping(p.memory.dcache.cache.homeCount, p.memory.dcache.cache.homeId)
  ))

  // The mesh sees the DDR in L2: the firmware and its data where they are, and above them three
  // windows that Loom points at the operands of a job: W, 512 MiB from 0xd0000000, then X and Z,
  // 128 MiB each from 0xf0000000 and 0xf8000000.
  def ddr(address: UInt): UInt =
    MuxCase(
      address,
      Seq(
        (address(31, 28) === "hd".U || address(31, 28) === "he".U) ->
          (loom.io.windows(0) + Cat(address(29), address(27, 0))),
        (address(31, 27) === "b11110".U)                           -> (loom.io.windows(1) + address(26, 0)),
        (address(31, 27) === "b11111".U)                           -> (loom.io.windows(2) + address(26, 0))
      )
    )

  dma.io.axi <> memory.io.matched
  dma.io.axi.ar.bits.addr := ddr(memory.io.matched.ar.bits.addr)
  dma.io.axi.aw.bits.addr := ddr(memory.io.matched.aw.bits.addr)
  hlink.memory.coherent(2) <> dma.io.coherent

  // The writes of the mesh to the DDR, for mtrace: channel 0 each burst at its DDR address
  // (write_mask: len, then size), channel 1 each beat (its strobe and data).
  val writes = Seq.fill(2)(Module(new MTraceDPI))
  writes.zipWithIndex.foreach { case (trace, channel) =>
    trace.io.clock     := clock
    trace.io.reset     := reset
    trace.io.is_write  := 1.U
    trace.io.is_shared := 0.U
    trace.io.channel   := channel.U
    trace.io.thread_id := 0.U
    trace.io.rob_id    := 0.U
    trace.io.inst_id   := 0.U
    trace.io.vbank_id  := 0.U
    trace.io.pbank_id  := 0.U
    trace.io.group_id  := 0.U
  }
  writes(0).io.addr := dma.io.axi.aw.bits.addr(31, 0)
  writes(0).io.write_mask := Cat(dma.io.axi.aw.bits.len, dma.io.axi.aw.bits.size)
  writes(0).io.data_lo    := 0.U
  writes(0).io.data_hi    := 0.U
  writes(0).io.enable     := dma.io.axi.aw.fire
  writes(1).io.addr       := 0.U
  writes(1).io.write_mask := dma.io.axi.w.bits.strb
  writes(1).io.data_lo    := dma.io.axi.w.bits.data(63, 0)
  writes(1).io.data_hi    := dma.io.axi.w.bits.data(127, 64)
  writes(1).io.enable     := dma.io.axi.w.fire

  // Mesh completion belongs to the hart; only Loom may exit the device process.
  hlink.memory.uncached <> loom.io.uncached
  val mailbox      = memory.io.other
  val output       = Instantiate(new Queue(UInt(8.W), 16))
  loom.io.output <> output.io.deq
  val writePending = RegInit(false.B)
  val replyPending = RegInit(false.B)
  val writeId      = Reg(UInt(p.hlink.uncached.idBits.W))
  val print        = Reg(Bool())
  val result       = Reg(UInt(32.W))
  val completed    = RegInit(false.B)
  mailbox.aw.ready                 := !writePending && !replyPending
  when(mailbox.aw.fire) {
    assert(
      mailbox.aw.bits.len === 0.U &&
        (mailbox.aw.bits.addr === "h60000000".U || mailbox.aw.bits.addr === "h60020000".U),
      "Invalid MAGIA mailbox address"
    )
    writeId      := mailbox.aw.bits.id
    print        := mailbox.aw.bits.addr === "h60020000".U
    writePending := true.B
  }
  mailbox.w.ready                  := writePending && (!print || output.io.enq.ready)
  output.io.enq.valid              := mailbox.w.valid && writePending && print
  output.io.enq.bits               := mailbox.w.bits.data(7, 0)
  when(mailbox.w.fire) {
    assert(mailbox.w.bits.last && mailbox.w.bits.strb(0), "Invalid MAGIA mailbox data")
    when(!print) {
      assert(mailbox.w.bits.strb(3, 0).andR, "MAGIA completion requires a word")
      result := mailbox.w.bits.data(31, 0)
    }
    writePending := false.B
    replyPending := true.B
  }
  mailbox.b.valid                  := replyPending
  mailbox.b.bits                   := 0.U.asTypeOf(mailbox.b.bits)
  mailbox.b.bits.id                := writeId
  when(mailbox.b.fire) {
    replyPending           := false.B
    when(!print)(completed := true.B)
  }
  mailbox.ar.ready                 := false.B
  mailbox.r.valid                  := false.B
  mailbox.r.bits                   := 0.U.asTypeOf(mailbox.r.bits)
  when(mailbox.ar.valid)(assert(false.B, "MAGIA completion is write-only"))
  when(!loom.io.running)(completed := false.B)

  // All harts publish their results before completion; retire external replies before resetting.
  val outstanding = RegInit(0.U(10.W))
  val accepted    = native.io.aw.fire.asUInt +& native.io.ar.fire.asUInt
  val retired     = native.io.b.fire.asUInt +& (native.io.r.fire && native.io.r.bits.last).asUInt
  outstanding       := outstanding + accepted - retired
  loom.io.completed := loom.io.running && completed && outstanding === 0.U &&
    !native.io.aw.valid && !native.io.ar.valid
  loom.io.result    := result

  hlink.memory.ddr.aw.valid := false.B
  hlink.memory.ddr.aw.bits  := 0.U.asTypeOf(hlink.memory.ddr.aw.bits)
  hlink.memory.ddr.w.valid  := false.B
  hlink.memory.ddr.w.bits   := 0.U.asTypeOf(hlink.memory.ddr.w.bits)
  hlink.memory.ddr.ar.valid := false.B
  hlink.memory.ddr.ar.bits  := 0.U.asTypeOf(hlink.memory.ddr.ar.bits)
  hlink.memory.ddr.b.ready  := true.B
  hlink.memory.ddr.r.ready  := true.B
  for ((up, down) <- Seq((native.io, memory.io.in))) {
    down.aw.valid := up.aw.valid
    up.aw.ready   := down.aw.ready
    down.ar.valid := up.ar.valid
    up.ar.ready   := down.ar.ready
    for ((dst, src) <- Seq((down.aw.bits, up.aw.bits), (down.ar.bits, up.ar.bits))) {
      dst       := 0.U.asTypeOf(dst)
      dst.id    := src.id; dst.addr    := src.addr; dst.len   := src.len; dst.size   := src.size
      dst.burst := src.burst; dst.lock := src.lock; dst.cache := src.cache; dst.prot := src.prot; dst.qos := src.qos
    }
    down.w <> up.w
    up.b <> down.b
    up.r <> down.r
  }
}

object MagiaComponent extends HartFactory {
  val component = "magia"

  def parameters(design: Design, source: HartPlacement): HartParam = {
    require(source.getImplementation.unpack(classOf[buckyball.components.magia.Params]).getProfile == "Mesh4x4")
    require(source.getHartId == 0 && source.getThreadIndicesCount == 0 && !source.hasAccelerator)
    val p = HartParam.fromProto(source, Seq.empty)
    p.copy(hlink =
      p.hlink.copy(
        backing = false,
        requesterIds = Seq(p.memory.icache.nodeId, p.memory.dcache.cache.nodeId, 3),
        ddr = p.memory.uncached.axi
      )
    )
  }

  def instantiate(
    design:   Design,
    source:   HartPlacement,
    hartp:    HartParam,
    simp:     SimParam,
    traceIds: Seq[Int]
  ): HLinkIO =
    Instantiate(new MagiaHart(
      hartp,
      FetchParam(source.getFetch.getInstructionBytes, source.getMemory.getIcache.getLines)
    )).hlink

}
