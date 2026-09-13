package memcore.memory.cache

import chisel3._
import chisel3.util._
import memcore.bus.chi._
import memcore.memory.coherence._

// Standalone multicore cache + Home + memory. Every protocol direction crosses
// a real CHI credit channel; the crossbar routes flits by destination NodeID.
class CoherentSystem(
  p:           ChiParams = ChiParams(),
  cores:       Int = 2,
  cacheLines:  Int = 4,
  memoryLines: Int = 64,
  creditDepth: Int = 4,
  npuCount:    Int = 0,
  homeCount:   Int = 2,
  cpuBanks:    Int = 2,
  npuSlots:    Int = 4)
    extends Module {

  val io = IO(new Bundle {
    val enable                  = Input(Bool())
    val bootReq                 = Flipped(Decoupled(new LineRequest(p)))
    val bootResp                = Decoupled(new LineResponse)
    val access                  = Vec(cores, Flipped(Decoupled(new CacheAccess(p))))
    val result                  = Vec(cores, Decoupled(new CacheResult))
    val hits                    = Output(Vec(cores, UInt(32.W)))
    val misses                  = Output(Vec(cores, UInt(32.W)))
    val busy                    = Output(Bool())
    val homeActive              = Output(UInt(homeCount.W))
    val parallelHomeCycles      = Output(UInt(32.W))
    val cancelledWritebackBeats = Output(UInt(32.W))
    val npuOutstanding          = Output(Vec(npuCount, UInt(32.W)))
    val cpuOutstanding          = Output(Vec(cores, UInt(32.W)))
    val npuCommand              = Vec(npuCount, Flipped(Decoupled(new NpuCommand(p))))
    val npuWrite                = Vec(npuCount, Flipped(Decoupled(new NpuBeat(p))))
    val npuRead                 = Vec(npuCount, Decoupled(new NpuBeat(p)))
    val npuDone                 = Vec(npuCount, Decoupled(Bool()))
    val regions                 = Output(Vec(npuCount, new RegionEntry(p)))
  })

  require(homeCount >= 1 && isPow2(homeCount) && memoryLines / homeCount >= 2)
  val mapping         = HomeMapping(homeCount)
  require(mapping.base + homeCount <= (1 << p.nodeIdBits))
  val caches          =
    Seq.tabulate(cores)(i => Module(new BankedChiCache(p, i + 1, cacheLines, homeCount = homeCount, banks = cpuBanks)))
  val npus            =
    Seq.tabulate(npuCount)(i => Module(new NpuRegionAgent(p, cores + i + 1, homeCount = homeCount, slots = npuSlots)))
  val agents          = cores + npuCount
  val regionDirectory = if (npuCount > 0) Some(Module(new RegionDirectory(p, npuCount, memoryLines))) else None

  val homes = Seq.tabulate(homeCount)(h =>
    Module(new ChiHome(p, agents, memoryLines, homeId = mapping.base + h, homeCount = homeCount, homeIndex = h))
  )

  val memories    = Seq.fill(homeCount)(Module(new ChiLineSram(p, memoryLines / homeCount)))
  val active      = RegNext(true.B, false.B)
  val enabled     = RegNext(io.enable, false.B)
  when(enabled)(assert(io.enable, "Coherent system must reset before entering boot mode"))
  val bootPending = RegInit(false.B)
  when(io.bootReq.fire)(bootPending  := true.B)
  when(io.bootResp.fire)(bootPending := false.B)
  when(io.enable)(assert(!bootPending, "Boot memory operation still pending at enable"))
  val bootHome = (io.bootReq.bits.addr >> 6) & (homeCount - 1).U
  io.bootReq.ready  := false.B
  io.bootResp.valid := !io.enable && memories.map(_.io.resp.valid).reduce(_ || _)
  io.bootResp.bits  := Mux1H(memories.map(m => m.io.resp.valid -> m.io.resp.bits))
  for (h <- 0 until homeCount) {
    val memory       = memories(h)
    val home         = homes(h)
    val bootSelected = bootHome === h.U && !bootPending
    memory.io.req.valid                      := Mux(io.enable, home.io.memoryReq.valid, io.bootReq.valid && bootSelected)
    memory.io.req.bits                       := Mux(io.enable, home.io.memoryReq.bits, io.bootReq.bits)
    when(!io.enable)(memory.io.req.bits.addr := mapping.localAddress(io.bootReq.bits.addr))
    home.io.memoryReq.ready                  := io.enable && memory.io.req.ready
    when(bootSelected)(io.bootReq.ready      := !io.enable && memory.io.req.ready)
    home.io.memoryResp.valid                 := io.enable && memory.io.resp.valid
    home.io.memoryResp.bits                  := memory.io.resp.bits
    memory.io.resp.ready                     := Mux(io.enable, home.io.memoryResp.ready, io.bootResp.ready)
  }
  when(!io.enable)(assert(PopCount(memories.map(_.io.resp.valid)) <= 1.U, "Multiple boot responses"))
  io.homeActive := VecInit(homes.map(_.io.busy)).asUInt
  io.busy := io.homeActive.orR
  val parallelCycles = RegInit(0.U(32.W))
  when(PopCount(io.homeActive) > 1.U)(parallelCycles := parallelCycles + 1.U)
  io.parallelHomeCycles                              := parallelCycles
  io.cancelledWritebackBeats                         := homes.map(_.io.cancelledWritebackBeats).reduce(_ + _)

  def channel[T <: ChiFlit](source: DecoupledIO[T], sink: DecoupledIO[T]): Unit = {
    val tx = Module(new ChiTx(source.bits.flitWidth, creditDepth))
    val rx = Module(new ChiRx(source.bits.flitWidth, creditDepth))
    tx.io.active    := active
    rx.io.active    := active
    tx.io.in.valid  := source.valid
    tx.io.in.bits   := source.bits.packed
    source.ready    := tx.io.in.ready
    rx.io.link <> tx.io.link
    sink.valid      := rx.io.out.valid
    sink.bits.unpack(rx.io.out.bits)
    rx.io.out.ready := sink.ready
  }

  // A combinational crossbar with one arbiter per destination. In particular,
  // there is NO queue after region admission: a lease blocks at Home acceptance.
  def route[T <: ChiFlit](inputs: Seq[DecoupledIO[T]], outputs: Seq[DecoupledIO[T]], destination: T => UInt): Unit = {
    val arbiters = outputs.map(out => Module(new RRArbiter(chiselTypeOf(out.bits), inputs.size)))
    outputs.zip(arbiters).foreach { case (out, arb) => out <> arb.io.out }
    for ((in, i) <- inputs.zipWithIndex) {
      val dest = destination(in.bits)
      in.ready := VecInit(arbiters.indices.map(d => dest === d.U && arbiters(d).io.in(i).ready)).asUInt.orR
      when(in.valid)(assert(dest < outputs.size.U, "Unroutable CHI NodeID"))
      for ((arb, d) <- arbiters.zipWithIndex) {
        arb.io.in(i).valid := in.valid && dest === d.U
        arb.io.in(i).bits  := in.bits
      }
    }
  }

  for (i <- 0 until cores) {
    val c = caches(i)
    c.io.access.valid    := io.enable && io.access(i).valid
    c.io.access.bits     := io.access(i).bits
    io.access(i).ready   := io.enable && c.io.access.ready
    io.result(i) <> c.io.result
    io.hits(i)           := c.io.hits
    io.misses(i)         := c.io.misses
    io.cpuOutstanding(i) := c.io.outstanding
  }
  for (i <- 0 until npuCount) {
    val n = npus(i)
    n.io.command.valid                := io.enable && io.npuCommand(i).valid
    n.io.command.bits                 := io.npuCommand(i).bits
    io.npuCommand(i).ready            := io.enable && n.io.command.ready
    n.io.write <> io.npuWrite(i)
    io.npuRead(i) <> n.io.read
    io.npuDone(i) <> n.io.done
    io.npuOutstanding(i)              := n.io.outstanding
    regionDirectory.get.io.claim(i) <> n.io.claim
    regionDirectory.get.io.publish(i) := n.io.publish
    regionDirectory.get.io.release(i) := n.io.release
    io.regions(i)                     := regionDirectory.get.io.entries(i)
  }
  val ports = caches.map(_.io.chi) ++ npus.map(_.io.chi)
  val requests        = Seq.fill(agents)(Wire(Decoupled(new ChiReq(p))))
  val responses       = Seq.fill(agents)(Wire(Decoupled(new ChiRsp(p))))
  val writeData       = Seq.fill(agents)(Wire(Decoupled(new ChiDat(p))))
  val returnResponses = Seq.fill(agents)(Wire(Decoupled(new ChiRsp(p))))
  val returnData      = Seq.fill(agents)(Wire(Decoupled(new ChiDat(p))))
  for (i <- 0 until agents) {
    val port  = ports(i)
    val retry = Module(new ChiRequestRetry(p, i + 1, records = if (i < cores) cpuBanks else npuSlots))
    retry.io.reqIn <> port.req
    retry.io.acceptedData.valid      := port.rxDat.fire && port.rxDat.bits.opcode === ChiOpcode.CompData.U
    retry.io.acceptedData.bits.srcId := port.rxDat.bits.srcId
    retry.io.acceptedData.bits.txnId := port.rxDat.bits.txnId
    val incoming   = Wire(Decoupled(new ChiReq(p)))
    channel(retry.io.reqOut, incoming)
    val release    = incoming.bits.opcode === ChiOpcode.Evict.U || incoming.bits.opcode === ChiOpcode.WriteBackFull.U
    val readShared = incoming.bits.opcode === ChiOpcode.ReadShared.U ||
      incoming.bits.opcode === ChiOpcode.ReadNotSharedDirty.U
    val blocked    =
      if (i < cores && npuCount > 0) {
        regionDirectory.get.io.entries.map(e =>
          e.valid && incoming.bits.addr >= e.base &&
            incoming.bits.addr < e.end && !release && !(e.published && !e.write && readShared)
        ).reduce(_ || _)
      } else false.B
    requests(i).valid := incoming.valid && !blocked
    requests(i).bits  := incoming.bits
    incoming.ready    := requests(i).ready && !blocked
    if (i >= cores) {
      val e = regionDirectory.get.io.entries(i - cores)
      when(incoming.valid) {
        assert(
          e.valid && incoming.bits.addr >= e.base && incoming.bits.addr < e.end,
          "NPU CHI request outside reservation"
        )
        when(incoming.bits.opcode === ChiOpcode.CleanInvalid.U) {
          assert(!e.published, "NPU cache sweep after lease publication")
        }.otherwise {
          assert(e.published, "NPU accessed data before CPU cache sweep completed")
          when(incoming.bits.opcode === ChiOpcode.WriteNoSnpPtl.U) {
            assert(e.write, "NPU write without exclusive lease")
          }
        }
      }
    }
    channel(port.txRsp, responses(i))
    channel(port.txDat, writeData(i))
    val snpArb = Module(new RRArbiter(new ChiSnp(p), homeCount))
    for (h <- 0 until homeCount) snpArb.io.in(h) <> homes(h).io.snp(i)
    channel(snpArb.io.out, port.snp)
    channel(returnResponses(i), retry.io.rspIn)
    port.rxRsp <> retry.io.rspOut
    channel(returnData(i), port.rxDat)
  }
  route(requests, homes.map(_.io.req), (r: ChiReq) => r.tgtId - mapping.base.U)
  route(responses, homes.map(_.io.rxRsp), (r: ChiRsp) => r.tgtId - mapping.base.U)
  route(writeData, homes.map(_.io.rxDat), (d: ChiDat) => d.tgtId - mapping.base.U)
  route(homes.map(_.io.rsp), returnResponses, (r: ChiRsp) => r.tgtId - 1.U)
  route(homes.map(_.io.dat), returnData, (d: ChiDat) => d.tgtId - 1.U)
  // Observe actual cache permissions, not just the Home's bookkeeping.
  // Unique excludes every other valid copy, including transiently arriving fills.
  for {
    a    <- 0 until cores
    b    <- a + 1 until cores
    x    <- 0 until cacheLines
    y    <- 0 until cacheLines
  } {
    val left  = caches(a).io.directory(x)
    val right = caches(b).io.directory(y)
    when(left.valid && right.valid && left.line === right.line) {
      assert(!left.writable && !right.writable, "Multiple copies coexist with Unique permission")
    }
  }
  for {
    n    <- 0 until npuCount
    c    <- 0 until cores
    line <- 0 until cacheLines
  } {
    val region  = regionDirectory.get.io.entries(n)
    val cached  = caches(c).io.directory(line)
    val address = cached.line << 6
    when(region.valid && region.published && cached.valid && address >= region.base && address < region.end) {
      assert(!region.write && !cached.writable, "CPU cache permission conflicts with published NPU lease")
    }
  }
}

object EmitCoherentSystem extends App {
  var dataBits  = 256
  var homeCount = 2

  val stageArgs = args.filterNot { arg =>
    if (arg.startsWith("--data-bits=")) { dataBits = arg.stripPrefix("--data-bits=").toInt; true }
    else if (arg.startsWith("--homes=")) { homeCount = arg.stripPrefix("--homes=").toInt; true }
    else false
  }

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new CoherentSystem(ChiParams(dataBits = dataBits), cores = 3, npuCount = 2, homeCount = homeCount),
    stageArgs
  )
}
