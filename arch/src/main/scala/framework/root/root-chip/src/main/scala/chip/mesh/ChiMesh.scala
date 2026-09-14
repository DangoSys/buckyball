package hier.chip.mesh

import chisel3._
import chisel3.util._
import memcore.bus.chi._

/**
 * CHI uses four non-overlapping Mesh VCs. A production tile mesh reserves at
 * least four more VCs for bulk DMA and future system traffic.
 */
object ChiMeshVirtualChannel {
  val Request  = 0
  val Response = 1
  val Data     = 2
  val Snoop    = 3
  val Required = 4
}

/**
 * Static placement of CHI NodeIDs in one mesh. Zero means "not present" only
 * when its coordinates are also zero; callers must validate their own node set.
 */
case class ChiMeshNodeMap(xByNode: Seq[Int], yByNode: Seq[Int], mesh: MeshParams) {
  require(xByNode.nonEmpty && xByNode.size == yByNode.size)
  require(xByNode.forall(x => x >= 0 && x < mesh.xNodes))
  require(yByNode.forall(y => y >= 0 && y < mesh.yNodes))
  def coordinateBits: Int = math.max(1, log2Ceil(xByNode.size))
}

/** A packed CHI flit with the routing destination kept outside protocol bits. */
class ChiMeshMessage(flitBits: Int, nodeIdBits: Int) extends Bundle {
  val targetNode = UInt(nodeIdBits.W)
  val flit       = UInt(flitBits.W)
}

/**
 * Turns one atomic CHI flit into a fixed-length wormhole packet.
 *
 * CHI channel identity is carried by the mesh VC, so a packetizer instance is
 * dedicated to one channel. `targetNode` is routing metadata, never serialized
 * into the CHI flit. This makes request, response, data and snoop packetizers
 * identical even though their target fields differ.
 */
class ChiMeshPacketizer(
  mesh:           MeshParams,
  flitBits:       Int,
  virtualChannel: Int,
  localX:         Int,
  localY:         Int,
  nodeMap:        ChiMeshNodeMap)
    extends Module {
  require(flitBits > 0)
  require(virtualChannel >= 0 && virtualChannel < mesh.virtualChannels)
  require(localX >= 0 && localX < mesh.xNodes && localY >= 0 && localY < mesh.yNodes)
  require(nodeMap.mesh == mesh)

  private val chunks     = (flitBits + mesh.payloadBits - 1) / mesh.payloadBits
  private val beatBits   = math.max(1, log2Ceil(chunks))
  private val paddedBits = chunks * mesh.payloadBits
  private val message    = new ChiMeshMessage(flitBits, nodeMap.coordinateBits)

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(message))
    val out = Decoupled(new MeshFlit(mesh))
  })

  val busy    = RegInit(false.B)
  val beat    = RegInit(0.U(beatBits.W))
  val flit    = Reg(UInt(flitBits.W))
  val target  = Reg(UInt(nodeMap.coordinateBits.W))
  val xLookup = VecInit(nodeMap.xByNode.map(_.U(mesh.xBits.W)))
  val yLookup = VecInit(nodeMap.yByNode.map(_.U(mesh.yBits.W)))
  val padded  = Wire(UInt(paddedBits.W))
  padded := flit

  io.in.ready := !busy
  when(io.in.fire) {
    assert(io.in.bits.targetNode < nodeMap.xByNode.size.U, "CHI Mesh target NodeID is not placed")
    flit   := io.in.bits.flit
    target := io.in.bits.targetNode
    beat   := 0.U
    busy   := true.B
  }

  io.out.valid        := busy
  io.out.bits.srcX    := localX.U
  io.out.bits.srcY    := localY.U
  io.out.bits.dstX    := xLookup(target)
  io.out.bits.dstY    := yLookup(target)
  io.out.bits.vc      := virtualChannel.U
  io.out.bits.head    := beat === 0.U
  io.out.bits.tail    := beat === (chunks - 1).U
  io.out.bits.payload := (padded >> (beat * mesh.payloadBits.U))(mesh.payloadBits - 1, 0)
  when(io.out.fire) {
    when(io.out.bits.tail) {
      busy := false.B
    }.otherwise {
      beat := beat + 1.U
    }
  }
}

/** Reassembles one fixed-length CHI packet after the mesh VC has been demultiplexed. */
class ChiMeshDepacketizer(mesh: MeshParams, flitBits: Int, virtualChannel: Int) extends Module {
  require(flitBits > 0)
  require(virtualChannel >= 0 && virtualChannel < mesh.virtualChannels)
  private val chunks     = (flitBits + mesh.payloadBits - 1) / mesh.payloadBits
  private val beatBits   = math.max(1, log2Ceil(chunks))
  private val packedBits = chunks * mesh.payloadBits

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new MeshFlit(mesh)))
    val out = Decoupled(UInt(flitBits.W))
  })

  val collecting  = RegInit(false.B)
  val beat        = RegInit(0.U(beatBits.W))
  val words       = Reg(Vec(chunks, UInt(mesh.payloadBits.W)))
  val result      = Reg(UInt(flitBits.W))
  val resultValid = RegInit(false.B)
  val assembled   = Wire(Vec(chunks, UInt(mesh.payloadBits.W)))
  for (index <- 0 until chunks) {
    assembled(index) := Mux(beat === index.U, io.in.bits.payload, words(index))
  }

  io.in.ready  := !resultValid
  io.out.valid := resultValid
  io.out.bits  := result
  when(io.in.fire) {
    assert(io.in.bits.vc === virtualChannel.U, "CHI packet arrived on wrong mesh VC")
    when(!collecting) {
      assert(io.in.bits.head, "CHI mesh packet started without head")
    }.otherwise {
      assert(!io.in.bits.head, "CHI mesh packet has a second head")
    }
    assert(io.in.bits.tail === (beat === (chunks - 1).U), "CHI mesh packet has wrong length")
    if (chunks == 1) {
      words(0) := io.in.bits.payload
    } else {
      words(beat) := io.in.bits.payload
    }
    when(io.in.bits.tail) {
      result      := Cat(assembled.reverse)(flitBits - 1, 0)
      resultValid := true.B
      collecting  := false.B
      beat        := 0.U
    }.otherwise {
      collecting := true.B
      beat       := beat + 1.U
    }
  }
  when(io.out.fire) {
    resultValid := false.B
  }
}

/**
 * One tile's CHI-facing Mesh endpoint.
 *
 * It owns packet boundaries and CHI channel-to-VC assignment. Its Mesh ports
 * carry opaque packets, so a router can be changed without changing cache or
 * Home protocol state. Snoop destinations are explicit sideband because CHI
 * Snp flits identify their source but do not contain a target NodeID.
 */
class ChiMeshEndpoint(
  p:       ChiParams,
  mesh:    MeshParams,
  localX:  Int,
  localY:  Int,
  nodeMap: ChiMeshNodeMap)
    extends Module {
  require(mesh.virtualChannels >= ChiMeshVirtualChannel.Required)
  require(nodeMap.mesh == mesh && nodeMap.coordinateBits == p.nodeIdBits)
  private val reqBits = (new ChiReq(p)).flitWidth
  private val rspBits = (new ChiRsp(p)).flitWidth
  private val datBits = (new ChiDat(p)).flitWidth
  private val snpBits = (new ChiSnp(p)).flitWidth
  private val message = (bits: Int) => new ChiMeshMessage(bits, p.nodeIdBits)

  val io = IO(new Bundle {
    val txReq   = Flipped(Decoupled(message(reqBits)))
    val txRsp   = Flipped(Decoupled(message(rspBits)))
    val txDat   = Flipped(Decoupled(message(datBits)))
    val txSnp   = Flipped(Decoupled(message(snpBits)))
    val rxReq   = Decoupled(UInt(reqBits.W))
    val rxRsp   = Decoupled(UInt(rspBits.W))
    val rxDat   = Decoupled(UInt(datBits.W))
    val rxSnp   = Decoupled(UInt(snpBits.W))
    val meshOut = Decoupled(new MeshFlit(mesh))
    val meshIn  = Flipped(Decoupled(new MeshFlit(mesh)))
  })

  val txReq   = Module(new ChiMeshPacketizer(mesh, reqBits, ChiMeshVirtualChannel.Request, localX, localY, nodeMap))
  val txRsp   = Module(new ChiMeshPacketizer(mesh, rspBits, ChiMeshVirtualChannel.Response, localX, localY, nodeMap))
  val txDat   = Module(new ChiMeshPacketizer(mesh, datBits, ChiMeshVirtualChannel.Data, localX, localY, nodeMap))
  val txSnp   = Module(new ChiMeshPacketizer(mesh, snpBits, ChiMeshVirtualChannel.Snoop, localX, localY, nodeMap))
  txReq.io.in <> io.txReq
  txRsp.io.in <> io.txRsp
  txDat.io.in <> io.txDat
  txSnp.io.in <> io.txSnp
  val arbiter = Module(new MeshPacketArbiter(mesh, ChiMeshVirtualChannel.Required))
  arbiter.io.in(ChiMeshVirtualChannel.Request) <> txReq.io.out
  arbiter.io.in(ChiMeshVirtualChannel.Response) <> txRsp.io.out
  arbiter.io.in(ChiMeshVirtualChannel.Data) <> txDat.io.out
  arbiter.io.in(ChiMeshVirtualChannel.Snoop) <> txSnp.io.out
  io.meshOut <> arbiter.io.out

  val rxReq     = Module(new ChiMeshDepacketizer(mesh, reqBits, ChiMeshVirtualChannel.Request))
  val rxRsp     = Module(new ChiMeshDepacketizer(mesh, rspBits, ChiMeshVirtualChannel.Response))
  val rxDat     = Module(new ChiMeshDepacketizer(mesh, datBits, ChiMeshVirtualChannel.Data))
  val rxSnp     = Module(new ChiMeshDepacketizer(mesh, snpBits, ChiMeshVirtualChannel.Snoop))
  val receivers = Seq(rxReq, rxRsp, rxDat, rxSnp)
  for ((receiver, vc) <- receivers.zipWithIndex) {
    receiver.io.in.valid := io.meshIn.valid && io.meshIn.bits.vc === vc.U
    receiver.io.in.bits  := io.meshIn.bits
  }
  io.meshIn.ready := MuxLookup(io.meshIn.bits.vc, false.B)(
    receivers.zipWithIndex.map { case (receiver, vc) => vc.U -> receiver.io.in.ready }
  )
  when(io.meshIn.valid) {
    assert(io.meshIn.bits.vc < ChiMeshVirtualChannel.Required.U, "Mesh packet is not a CHI endpoint VC")
  }
  io.rxReq <> rxReq.io.out
  io.rxRsp <> rxRsp.io.out
  io.rxDat <> rxDat.io.out
  io.rxSnp <> rxSnp.io.out
}

/** Typed adapter for a cache or NPU CHI requester port. */
class ChiMeshRequesterEndpoint(
  p:       ChiParams,
  mesh:    MeshParams,
  localX:  Int,
  localY:  Int,
  nodeMap: ChiMeshNodeMap)
    extends Module {

  val io = IO(new Bundle {
    val chi     = Flipped(new memcore.memory.coherence.ChiRequesterPort(p))
    val meshOut = Decoupled(new MeshFlit(mesh))
    val meshIn  = Flipped(Decoupled(new MeshFlit(mesh)))
  })

  val endpoint = Module(new ChiMeshEndpoint(p, mesh, localX, localY, nodeMap))
  endpoint.io.txReq.valid           := io.chi.req.valid
  endpoint.io.txReq.bits.targetNode := io.chi.req.bits.tgtId
  endpoint.io.txReq.bits.flit       := io.chi.req.bits.packed
  io.chi.req.ready                  := endpoint.io.txReq.ready
  endpoint.io.txRsp.valid           := io.chi.txRsp.valid
  endpoint.io.txRsp.bits.targetNode := io.chi.txRsp.bits.tgtId
  endpoint.io.txRsp.bits.flit       := io.chi.txRsp.bits.packed
  io.chi.txRsp.ready                := endpoint.io.txRsp.ready
  endpoint.io.txDat.valid           := io.chi.txDat.valid
  endpoint.io.txDat.bits.targetNode := io.chi.txDat.bits.tgtId
  endpoint.io.txDat.bits.flit       := io.chi.txDat.bits.packed
  io.chi.txDat.ready                := endpoint.io.txDat.ready
  endpoint.io.txSnp.valid           := false.B
  endpoint.io.txSnp.bits            := 0.U.asTypeOf(endpoint.io.txSnp.bits)
  endpoint.io.rxRsp.ready           := io.chi.rxRsp.ready
  io.chi.rxRsp.valid                := endpoint.io.rxRsp.valid
  io.chi.rxRsp.bits.unpack(endpoint.io.rxRsp.bits)
  endpoint.io.rxDat.ready           := io.chi.rxDat.ready
  io.chi.rxDat.valid                := endpoint.io.rxDat.valid
  io.chi.rxDat.bits.unpack(endpoint.io.rxDat.bits)
  endpoint.io.rxSnp.ready           := io.chi.snp.ready
  io.chi.snp.valid                  := endpoint.io.rxSnp.valid
  io.chi.snp.bits.unpack(endpoint.io.rxSnp.bits)
  endpoint.io.rxReq.ready           := true.B
  io.meshOut <> endpoint.io.meshOut
  endpoint.io.meshIn <> io.meshIn
}

/** Typed adapter for one directory Home and all of its snoop targets. */
class ChiMeshHomeEndpoint(
  p:       ChiParams,
  mesh:    MeshParams,
  localX:  Int,
  localY:  Int,
  nodeMap: ChiMeshNodeMap,
  agents:  Int)
    extends Module {
  require(agents >= 1 && agents < (1 << p.nodeIdBits))
  private val snpBits = (new ChiSnp(p)).flitWidth

  val io = IO(new Bundle {
    val req     = Decoupled(new ChiReq(p))
    val rxRsp   = Decoupled(new ChiRsp(p))
    val rxDat   = Decoupled(new ChiDat(p))
    val rsp     = Flipped(Decoupled(new ChiRsp(p)))
    val dat     = Flipped(Decoupled(new ChiDat(p)))
    val snp     = Vec(agents, Flipped(Decoupled(new ChiSnp(p))))
    val meshOut = Decoupled(new MeshFlit(mesh))
    val meshIn  = Flipped(Decoupled(new MeshFlit(mesh)))
  })

  val endpoint = Module(new ChiMeshEndpoint(p, mesh, localX, localY, nodeMap))
  endpoint.io.txReq.valid           := false.B
  endpoint.io.txReq.bits            := 0.U.asTypeOf(endpoint.io.txReq.bits)
  endpoint.io.txRsp.valid           := io.rsp.valid
  endpoint.io.txRsp.bits.targetNode := io.rsp.bits.tgtId
  endpoint.io.txRsp.bits.flit       := io.rsp.bits.packed
  io.rsp.ready                      := endpoint.io.txRsp.ready
  endpoint.io.txDat.valid           := io.dat.valid
  endpoint.io.txDat.bits.targetNode := io.dat.bits.tgtId
  endpoint.io.txDat.bits.flit       := io.dat.bits.packed
  io.dat.ready                      := endpoint.io.txDat.ready
  val snpArbiter = Module(new RRArbiter(new ChiMeshMessage(snpBits, p.nodeIdBits), agents))
  for (agent <- 0 until agents) {
    snpArbiter.io.in(agent).valid           := io.snp(agent).valid
    snpArbiter.io.in(agent).bits.targetNode := (agent + 1).U
    snpArbiter.io.in(agent).bits.flit       := io.snp(agent).bits.packed
    io.snp(agent).ready                     := snpArbiter.io.in(agent).ready
  }
  endpoint.io.txSnp <> snpArbiter.io.out
  endpoint.io.rxReq.ready := io.req.ready
  io.req.valid            := endpoint.io.rxReq.valid
  io.req.bits.unpack(endpoint.io.rxReq.bits)
  endpoint.io.rxRsp.ready := io.rxRsp.ready
  io.rxRsp.valid          := endpoint.io.rxRsp.valid
  io.rxRsp.bits.unpack(endpoint.io.rxRsp.bits)
  endpoint.io.rxDat.ready := io.rxDat.ready
  io.rxDat.valid          := endpoint.io.rxDat.valid
  io.rxDat.bits.unpack(endpoint.io.rxDat.bits)
  endpoint.io.rxSnp.ready := true.B
  when(endpoint.io.rxSnp.valid)(assert(false.B, "Home received a CHI snoop"))
  io.meshOut <> endpoint.io.meshOut
  endpoint.io.meshIn <> io.meshIn
}

/** Native verification target for Home snoop target-sideband routing. */
class ChiMeshHomeSnoopLoopback extends Module {
  private val chi  = ChiParams()
  private val mesh = MeshParams(xNodes = 2, yNodes = 1, payloadBits = 32, virtualChannels = 8)

  private val map = ChiMeshNodeMap(
    Seq.tabulate(1 << chi.nodeIdBits)(node => if (node == 1) 1 else 0),
    Seq.fill(1 << chi.nodeIdBits)(0),
    mesh
  )

  private val snpBits = (new ChiSnp(chi)).flitWidth

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(UInt(snpBits.W)))
    val out = Decoupled(UInt(snpBits.W))
  })

  val home     = Module(new ChiMeshHomeEndpoint(chi, mesh, localX = 0, localY = 0, nodeMap = map, agents = 2))
  val receiver = Module(new ChiMeshEndpoint(chi, mesh, localX = 1, localY = 0, nodeMap = map))
  home.io.snp(0).valid      := io.in.valid
  home.io.snp(0).bits.unpack(io.in.bits)
  io.in.ready               := home.io.snp(0).ready
  home.io.snp(1).valid      := false.B
  home.io.snp(1).bits       := 0.U.asTypeOf(home.io.snp(1).bits)
  home.io.rsp.valid         := false.B
  home.io.rsp.bits          := 0.U.asTypeOf(home.io.rsp.bits)
  home.io.dat.valid         := false.B
  home.io.dat.bits          := 0.U.asTypeOf(home.io.dat.bits)
  home.io.req.ready         := true.B
  home.io.rxRsp.ready       := true.B
  home.io.rxDat.ready       := true.B
  receiver.io.txReq.valid   := false.B
  receiver.io.txReq.bits    := 0.U.asTypeOf(receiver.io.txReq.bits)
  receiver.io.txRsp.valid   := false.B
  receiver.io.txRsp.bits    := 0.U.asTypeOf(receiver.io.txRsp.bits)
  receiver.io.txDat.valid   := false.B
  receiver.io.txDat.bits    := 0.U.asTypeOf(receiver.io.txDat.bits)
  receiver.io.txSnp.valid   := false.B
  receiver.io.txSnp.bits    := 0.U.asTypeOf(receiver.io.txSnp.bits)
  receiver.io.rxReq.ready   := true.B
  receiver.io.rxRsp.ready   := true.B
  receiver.io.rxDat.ready   := true.B
  io.out <> receiver.io.rxSnp
  receiver.io.meshIn.valid  := home.io.meshOut.valid
  receiver.io.meshIn.bits   := home.io.meshOut.bits
  home.io.meshOut.ready     := receiver.io.meshIn.ready
  home.io.meshIn.valid      := false.B
  home.io.meshIn.bits       := 0.U.asTypeOf(home.io.meshIn.bits)
  receiver.io.meshOut.ready := true.B
}

object EmitChiMeshHomeSnoopLoopback extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(new ChiMeshHomeSnoopLoopback, args)
}

/** Native endpoint verification target: a real CHI Dat channel traverses its Mesh endpoint. */
class ChiMeshEndpointLoopback extends Module {
  private val chi  = ChiParams()
  private val mesh = MeshParams(xNodes = 2, yNodes = 2, payloadBits = 32, virtualChannels = 8)

  private val map = ChiMeshNodeMap(
    Seq.tabulate(1 << chi.nodeIdBits)(_ & 1),
    Seq.tabulate(1 << chi.nodeIdBits)(node => (node >> 1) & 1),
    mesh
  )

  private val datBits = (new ChiDat(chi)).flitWidth

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new ChiMeshMessage(datBits, chi.nodeIdBits)))
    val out = Decoupled(UInt(datBits.W))
  })

  val endpoint = Module(new ChiMeshEndpoint(chi, mesh, localX = 0, localY = 0, nodeMap = map))
  endpoint.io.txDat <> io.in
  endpoint.io.txReq.valid   := false.B
  endpoint.io.txReq.bits    := 0.U.asTypeOf(endpoint.io.txReq.bits)
  endpoint.io.txRsp.valid   := false.B
  endpoint.io.txRsp.bits    := 0.U.asTypeOf(endpoint.io.txRsp.bits)
  endpoint.io.txSnp.valid   := false.B
  endpoint.io.txSnp.bits    := 0.U.asTypeOf(endpoint.io.txSnp.bits)
  endpoint.io.rxReq.ready   := true.B
  endpoint.io.rxRsp.ready   := true.B
  endpoint.io.rxSnp.ready   := true.B
  io.out <> endpoint.io.rxDat
  endpoint.io.meshIn.valid  := endpoint.io.meshOut.valid
  endpoint.io.meshIn.bits   := endpoint.io.meshOut.bits
  endpoint.io.meshOut.ready := endpoint.io.meshIn.ready
}

object EmitChiMeshEndpointLoopback extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(new ChiMeshEndpointLoopback, args)
}

/** Native verification target for packetization and reassembly without a mesh topology. */
class ChiMeshCodecLoopback extends Module {
  private val mesh = MeshParams(xNodes = 2, yNodes = 2, payloadBits = 32, virtualChannels = 4)
  private val chi  = ChiParams()

  private val map = ChiMeshNodeMap(
    Seq.tabulate(1 << chi.nodeIdBits)(_ & 1),
    Seq.tabulate(1 << chi.nodeIdBits)(node => (node >> 1) & 1),
    mesh
  )

  private val flitBits = (new ChiDat(chi)).flitWidth

  val io = IO(new Bundle {
    val in            = Flipped(Decoupled(new ChiMeshMessage(flitBits, map.coordinateBits)))
    val out           = Decoupled(UInt(flitBits.W))
    val observedValid = Output(Bool())
    val observed      = Output(new MeshFlit(mesh))
  })

  val tx = Module(new ChiMeshPacketizer(
    mesh,
    flitBits,
    virtualChannel = ChiMeshVirtualChannel.Data,
    localX = 0,
    localY = 0,
    nodeMap = map
  ))

  val rx = Module(new ChiMeshDepacketizer(mesh, flitBits, virtualChannel = ChiMeshVirtualChannel.Data))
  io.in <> tx.io.in
  rx.io.in <> tx.io.out
  io.out <> rx.io.out
  io.observedValid := tx.io.out.valid
  io.observed      := tx.io.out.bits
}

object EmitChiMeshCodecLoopback extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(new ChiMeshCodecLoopback, args)
}
