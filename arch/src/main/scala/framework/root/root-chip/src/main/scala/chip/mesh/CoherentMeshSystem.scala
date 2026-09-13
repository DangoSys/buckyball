package hier.chip.mesh

import chisel3._
import chisel3.util._
import memcore.bus.chi._
import memcore.memory.coherence._
import memcore.memory.cache.BankedChiCache

/**
 * Standalone coherent system using CHI-over-credit-Mesh.
 *
 * Requester Nodes occupy the first local mesh ports. Homes follow them. The
 * cache and Home state machines are unchanged; only their former crossbar
 * connection is replaced by typed Mesh endpoints.
 */
class CoherentMeshSystem(
  p:               ChiParams = ChiParams(),
  cacheLines:      Int = 4,
  memoryLines:     Int = 64,
  cpuBanks:        Int = 2,
  linkDepth:       Int = 2,
  cores:           Int = 2,
  homeCount:       Int = 2,
  meshXNodes:      Int = 2,
  npuCount:        Int = 0,
  npuSlots:        Int = 4,
  meshPayloadBits: Int = 320)
    extends Module {
  require(cores >= 2 && cores < 64)
  require(npuCount >= 0 && cores + npuCount < 64)
  require(homeCount >= 1 && isPow2(homeCount) && memoryLines >= 2 * homeCount && memoryLines % homeCount == 0)
  require(meshXNodes >= 1 && isPow2(meshXNodes))
  require(meshPayloadBits >= 320 && meshPayloadBits                                          % 8 == 0)
  private val agents     = cores + npuCount
  private val tiles      = agents + homeCount
  private val meshYNodes = (tiles + meshXNodes - 1) / meshXNodes
  private val mapping    = HomeMapping(homeCount)
  require(mapping.base + homeCount <= (1 << p.nodeIdBits))
  private val meshParams =
    MeshParams(xNodes = meshXNodes, yNodes = meshYNodes, payloadBits = meshPayloadBits, virtualChannels = 8)

  private def slot(node: Int): Int =
    if (node >= 1 && node <= agents) node - 1
    else if (node >= mapping.base && node < mapping.base + homeCount) agents + node - mapping.base
    else 0

  private val xByNode = Seq.tabulate(1 << p.nodeIdBits)(node => slot(node) % meshXNodes)
  private val yByNode = Seq.tabulate(1 << p.nodeIdBits)(node => slot(node) / meshXNodes)
  private val nodeMap = ChiMeshNodeMap(xByNode, yByNode, meshParams)

  val io = IO(new Bundle {
    val enable     = Input(Bool())
    val bootReq    = Flipped(Decoupled(new LineRequest(p)))
    val bootResp   = Decoupled(new LineResponse)
    val access     = Vec(cores, Flipped(Decoupled(new CacheAccess(p))))
    val result     = Vec(cores, Decoupled(new CacheResult))
    val hits       = Output(Vec(cores, UInt(32.W)))
    val misses     = Output(Vec(cores, UInt(32.W)))
    val npuCommand = Vec(npuCount, Flipped(Decoupled(new NpuCommand(p))))
    val npuWrite   = Vec(npuCount, Flipped(Decoupled(new NpuBeat(p))))
    val npuRead    = Vec(npuCount, Decoupled(new NpuBeat(p)))
    val npuDone    = Vec(npuCount, Decoupled(Bool()))
    val regions    = Output(Vec(npuCount, new RegionEntry(p)))
    val busy       = Output(Bool())
  })

  val caches = Seq.tabulate(cores)(index =>
    Module(new BankedChiCache(p, index + 1, cacheLines, homeCount = homeCount, banks = cpuBanks))
  )

  val npus = Seq.tabulate(npuCount)(index =>
    Module(new NpuRegionAgent(p, cores + index + 1, homeCount = homeCount, slots = npuSlots))
  )

  val regionDirectory = if (npuCount > 0) Some(Module(new RegionDirectory(p, npuCount, memoryLines))) else None

  val homeNodes = Seq.tabulate(homeCount)(index =>
    Module(new ChiHome(
      p,
      agents = agents,
      lines = memoryLines,
      homeId = mapping.base + index,
      homeCount = homeCount,
      homeIndex = index
    ))
  )

  val memories = Seq.fill(homeCount)(Module(new ChiLineSram(p, memoryLines / homeCount)))
  val mesh     = Module(new MeshCreditNetwork(meshParams, linkDepth))
  mesh.io.active := true.B
  for (port <- tiles until meshParams.xNodes * meshParams.yNodes) {
    mesh.io.localIn(port).valid  := false.B
    mesh.io.localIn(port).bits   := 0.U.asTypeOf(mesh.io.localIn(port).bits)
    mesh.io.localOut(port).ready := true.B
  }
  private def xOf(port: Int): Int = port % meshXNodes
  private def yOf(port: Int): Int = port / meshXNodes

  val cacheEndpoints = Seq.tabulate(cores)(index =>
    Module(new ChiMeshRequesterEndpoint(p, meshParams, localX = xOf(index), localY = yOf(index), nodeMap))
  )

  val npuEndpoints = Seq.tabulate(npuCount)(index =>
    Module(new TileMeshEndpoint(p, meshParams, localX = xOf(cores + index), localY = yOf(cores + index), nodeMap))
  )

  val homeEndpoints = Seq.tabulate(homeCount)(index =>
    Module(new ChiMeshHomeEndpoint(
      p,
      meshParams,
      localX = xOf(agents + index),
      localY = yOf(agents + index),
      nodeMap,
      agents = agents
    ))
  )

  for (index <- 0 until cores) {
    cacheEndpoints(index).io.chi <> caches(index).io.chi
    mesh.io.localIn(index) <> cacheEndpoints(index).io.meshOut
    cacheEndpoints(index).io.meshIn <> mesh.io.localOut(index)
    caches(index).io.access.valid := io.enable && io.access(index).valid
    caches(index).io.access.bits  := io.access(index).bits
    io.access(index).ready        := io.enable && caches(index).io.access.ready
    io.result(index) <> caches(index).io.result
    io.hits(index)                := caches(index).io.hits
    io.misses(index)              := caches(index).io.misses
  }
  for (index <- 0 until npuCount) {
    val npu      = npus(index)
    val endpoint = npuEndpoints(index)
    endpoint.io.chi <> npu.io.chi
    endpoint.io.bulkIn.valid              := false.B
    endpoint.io.bulkIn.bits               := 0.U.asTypeOf(endpoint.io.bulkIn.bits)
    endpoint.io.bulkOut.ready             := true.B
    mesh.io.localIn(cores + index) <> endpoint.io.meshOut
    endpoint.io.meshIn <> mesh.io.localOut(cores + index)
    npu.io.command.valid                  := io.enable && io.npuCommand(index).valid
    npu.io.command.bits                   := io.npuCommand(index).bits
    io.npuCommand(index).ready            := io.enable && npu.io.command.ready
    npu.io.write <> io.npuWrite(index)
    io.npuRead(index) <> npu.io.read
    io.npuDone(index) <> npu.io.done
    regionDirectory.get.io.claim(index) <> npu.io.claim
    regionDirectory.get.io.publish(index) := npu.io.publish
    regionDirectory.get.io.release(index) := npu.io.release
    io.regions(index)                     := regionDirectory.get.io.entries(index)
  }
  for (index <- 0 until homeCount) {
    val home       = homeNodes(index)
    val endpoint   = homeEndpoints(index)
    val incoming   = Wire(Decoupled(new ChiReq(p)))
    incoming <> endpoint.io.req
    val release    = incoming.bits.opcode === ChiOpcode.Evict.U || incoming.bits.opcode === ChiOpcode.WriteBackFull.U
    val readShared = incoming.bits.opcode === ChiOpcode.ReadShared.U ||
      incoming.bits.opcode === ChiOpcode.ReadNotSharedDirty.U
    val blocked    =
      if (npuCount > 0) {
        regionDirectory.get.io.entries.map(entry =>
          incoming.bits.srcId <= cores.U && entry.valid && incoming.bits.addr >= entry.base &&
            incoming.bits.addr < entry.end && !release && !(entry.published && !entry.write && readShared)
        ).reduce(_ || _)
      } else false.B
    home.io.req.valid := incoming.valid && !blocked
    home.io.req.bits  := incoming.bits
    incoming.ready    := home.io.req.ready && !blocked
    if (npuCount > 0) {
      for (index <- 0 until npuCount) {
        val entry = regionDirectory.get.io.entries(index)
        when(incoming.valid && incoming.bits.srcId === (cores + index + 1).U) {
          assert(
            entry.valid && incoming.bits.addr >= entry.base && incoming.bits.addr < entry.end,
            "NPU CHI request outside reservation"
          )
          when(incoming.bits.opcode === ChiOpcode.CleanInvalid.U) {
            assert(!entry.published, "NPU cache sweep after lease publication")
          }.otherwise {
            assert(entry.published, "NPU accessed data before CPU cache sweep completed")
            when(incoming.bits.opcode === ChiOpcode.WriteNoSnpPtl.U) {
              assert(entry.write, "NPU write without exclusive lease")
            }
          }
        }
      }
    }
    home.io.rxRsp <> endpoint.io.rxRsp
    home.io.rxDat <> endpoint.io.rxDat
    endpoint.io.rsp <> home.io.rsp
    endpoint.io.dat <> home.io.dat
    for (agent <- 0 until agents) home.io.snp(agent) <> endpoint.io.snp(agent)
    mesh.io.localIn(index + agents) <> endpoint.io.meshOut
    endpoint.io.meshIn <> mesh.io.localOut(index + agents)
  }

  val bootPending = RegInit(false.B)
  when(io.bootReq.fire)(bootPending  := true.B)
  when(io.bootResp.fire)(bootPending := false.B)
  when(io.enable)(assert(!bootPending, "Boot memory operation still pending at enable"))
  val bootHome = (io.bootReq.bits.addr >> 6) & (homeCount - 1).U
  io.bootReq.ready  := false.B
  io.bootResp.valid := !io.enable && memories.map(_.io.resp.valid).reduce(_ || _)
  io.bootResp.bits  := Mux1H(memories.map(memory => memory.io.resp.valid -> memory.io.resp.bits))
  for (index <- 0 until homeCount) {
    val memory   = memories(index)
    val home     = homeNodes(index)
    val selected = bootHome === index.U && !bootPending
    memory.io.req.valid                      := Mux(io.enable, home.io.memoryReq.valid, io.bootReq.valid && selected)
    memory.io.req.bits                       := Mux(io.enable, home.io.memoryReq.bits, io.bootReq.bits)
    when(!io.enable)(memory.io.req.bits.addr := mapping.localAddress(io.bootReq.bits.addr))
    home.io.memoryReq.ready                  := io.enable && memory.io.req.ready
    when(selected)(io.bootReq.ready          := !io.enable && memory.io.req.ready)
    home.io.memoryResp.valid                 := io.enable && memory.io.resp.valid
    home.io.memoryResp.bits                  := memory.io.resp.bits
    memory.io.resp.ready                     := Mux(io.enable, home.io.memoryResp.ready, io.bootResp.ready)
  }
  when(!io.enable)(assert(PopCount(memories.map(_.io.resp.valid)) <= 1.U, "Multiple boot responses"))
  io.busy := homeNodes.map(_.io.busy).reduce(_ || _)
}

object EmitCoherentMeshSystem extends App {
  var cores       = 2
  var homes       = 2
  var meshX       = 2
  var npus        = 0
  var payloadBits = 320

  val stageArgs = args.filterNot { arg =>
    if (arg.startsWith("--cores=")) { cores = arg.stripPrefix("--cores=").toInt; true }
    else if (arg.startsWith("--homes=")) { homes = arg.stripPrefix("--homes=").toInt; true }
    else if (arg.startsWith("--mesh-x=")) { meshX = arg.stripPrefix("--mesh-x=").toInt; true }
    else if (arg.startsWith("--npus=")) { npus = arg.stripPrefix("--npus=").toInt; true }
    else if (arg.startsWith("--mesh-payload-bits=")) {
      payloadBits = arg.stripPrefix("--mesh-payload-bits=").toInt; true
    } else false
  }

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new CoherentMeshSystem(
      cores = cores,
      homeCount = homes,
      meshXNodes = meshX,
      npuCount = npus,
      meshPayloadBits = payloadBits
    ),
    stageArgs
  )
}
