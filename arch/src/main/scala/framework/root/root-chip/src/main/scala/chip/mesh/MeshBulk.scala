package hier.chip.mesh

import chisel3._
import chisel3.util._
import memcore.bus.axi.Beat

/** One AXI-stream beat plus the Mesh destination selected at packet start. */
class MeshBulkBeat(dataBits: Int, nodeIdBits: Int) extends Bundle {
  require(dataBits > 0 && dataBits % 8 == 0)
  val targetNode = UInt(nodeIdBits.W)
  val data       = UInt(dataBits.W)
  val keep       = UInt((dataBits / 8).W)
  val last       = Bool()
}

/** VC5 control descriptor for one remote scratchpad DMA transaction. */
class BulkDescriptor(nodeIdBits: Int, addressBits: Int = 32) extends Bundle {
  val sourceNode     = UInt(nodeIdBits.W)
  val targetNode     = UInt(nodeIdBits.W)
  val write          = Bool()
  val scratchpadAddr = UInt(addressBits.W)
  val bytes          = UInt(32.W)
  val txnId          = UInt(8.W)
}

/** VC6 acknowledgement or terminal completion for a bulk descriptor. */
class BulkCompletion(nodeIdBits: Int) extends Bundle {
  val targetNode = UInt(nodeIdBits.W)
  val txnId      = UInt(8.W)
  val ready      = Bool()
  val error      = Bool()
}

/** One-flit control packet for VC5 descriptors and VC6 acknowledgements. */
class MeshBulkControlTx(
  mesh:           MeshParams,
  payloadBits:    Int,
  virtualChannel: Int,
  localX:         Int,
  localY:         Int,
  nodeMap:        ChiMeshNodeMap)
    extends Module {
  require(payloadBits > 0 && payloadBits <= mesh.payloadBits)

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new MeshBulkBeat(payloadBits, nodeMap.coordinateBits)))
    val out = Decoupled(new MeshFlit(mesh))
  })

  val xLookup = VecInit(nodeMap.xByNode.map(_.U(mesh.xBits.W)))
  val yLookup = VecInit(nodeMap.yByNode.map(_.U(mesh.yBits.W)))
  io.out.valid        := io.in.valid
  io.out.bits.srcX    := localX.U
  io.out.bits.srcY    := localY.U
  io.out.bits.dstX    := xLookup(io.in.bits.targetNode)
  io.out.bits.dstY    := yLookup(io.in.bits.targetNode)
  io.out.bits.vc      := virtualChannel.U
  io.out.bits.head    := true.B
  io.out.bits.tail    := true.B
  io.out.bits.payload := io.in.bits.data
  io.in.ready         := io.out.ready
}

class MeshBulkControlRx(mesh: MeshParams, payloadBits: Int, virtualChannel: Int) extends Module {
  require(payloadBits > 0 && payloadBits <= mesh.payloadBits)

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new MeshFlit(mesh)))
    val out = Decoupled(UInt(payloadBits.W))
  })

  io.out.valid := io.in.valid
  io.out.bits  := io.in.bits.payload(payloadBits - 1, 0)
  io.in.ready  := io.out.ready
  when(io.in.fire) {
    assert(
      io.in.bits.vc === virtualChannel.U && io.in.bits.head && io.in.bits.tail,
      "Bulk control packet must be one flit on its assigned VC"
    )
  }
}

/**
 * Maps one AXI-stream transaction to one non-interleaved Mesh packet.
 *
 * Mesh `head` is the first AXI beat and `tail` is the beat carrying AXI
 * `last`. The Mesh router therefore holds the selected output across the full
 * long stream, rather than arbitrating again at every beat.
 */
class MeshBulkPacketizer(
  mesh:           MeshParams,
  dataBits:       Int,
  virtualChannel: Int,
  localX:         Int,
  localY:         Int,
  nodeMap:        ChiMeshNodeMap)
    extends Module {
  require(mesh.payloadBits >= dataBits + dataBits / 8)
  require(virtualChannel >= 0 && virtualChannel < mesh.virtualChannels)
  require(nodeMap.mesh == mesh)

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new MeshBulkBeat(dataBits, nodeMap.coordinateBits)))
    val out = Decoupled(new MeshFlit(mesh))
  })

  val first          = RegInit(true.B)
  val target         = Reg(UInt(nodeMap.coordinateBits.W))
  val xLookup        = VecInit(nodeMap.xByNode.map(_.U(mesh.xBits.W)))
  val yLookup        = VecInit(nodeMap.yByNode.map(_.U(mesh.yBits.W)))
  val selectedTarget = Mux(first, io.in.bits.targetNode, target)
  val payload        = Wire(UInt(mesh.payloadBits.W))
  payload             := Cat(io.in.bits.keep, io.in.bits.data)
  io.out.valid        := io.in.valid
  io.out.bits.srcX    := localX.U
  io.out.bits.srcY    := localY.U
  io.out.bits.dstX    := xLookup(selectedTarget)
  io.out.bits.dstY    := yLookup(selectedTarget)
  io.out.bits.vc      := virtualChannel.U
  io.out.bits.head    := first
  io.out.bits.tail    := io.in.bits.last
  io.out.bits.payload := payload
  io.in.ready         := io.out.ready
  when(io.in.fire) {
    assert(io.in.bits.targetNode < nodeMap.xByNode.size.U, "Bulk Mesh target NodeID is not placed")
    when(!first)(assert(io.in.bits.targetNode === target, "Bulk Mesh target changed inside AXI packet"))
    target := io.in.bits.targetNode
    first  := io.in.bits.last
  }
}

/** Recreates AXI-stream beat boundaries from a single bulk Mesh VC. */
class MeshBulkDepacketizer(mesh: MeshParams, dataBits: Int, virtualChannel: Int) extends Module {
  require(mesh.payloadBits >= dataBits + dataBits / 8)
  require(virtualChannel >= 0 && virtualChannel < mesh.virtualChannels)

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new MeshFlit(mesh)))
    val out = Decoupled(new Beat(dataBits))
  })

  val first = RegInit(true.B)
  io.out.valid     := io.in.valid
  io.out.bits.data := io.in.bits.payload(dataBits - 1, 0)
  io.out.bits.keep := io.in.bits.payload(dataBits + dataBits / 8 - 1, dataBits)
  io.out.bits.last := io.in.bits.tail
  io.out.bits.id   := 0.U
  io.out.bits.dest := 0.U
  io.out.bits.user := 0.U
  io.in.ready      := io.out.ready
  when(io.in.fire) {
    assert(io.in.bits.vc === virtualChannel.U, "Bulk packet arrived on wrong Mesh VC")
    assert(io.in.bits.head === first, "Bulk Mesh packet head mismatch")
    first := io.in.bits.tail
  }
}

/** Native verification target for long AXI-stream packet ownership over Mesh. */
class MeshBulkLoopback extends Module {
  private val mesh = MeshParams(xNodes = 2, yNodes = 2, payloadBits = 320, virtualChannels = 8)
  private val map  = ChiMeshNodeMap(Seq.tabulate(128)(_ & 1), Seq.tabulate(128)(node => (node >> 1) & 1), mesh)

  val io = IO(new Bundle {
    val in            = Flipped(Decoupled(new MeshBulkBeat(256, 7)))
    val out           = Decoupled(new Beat(256))
    val observed      = Output(new MeshFlit(mesh))
    val observedValid = Output(Bool())
  })

  val tx = Module(new MeshBulkPacketizer(mesh, 256, virtualChannel = 4, localX = 0, localY = 0, map))
  val rx = Module(new MeshBulkDepacketizer(mesh, 256, virtualChannel = 4))
  tx.io.in <> io.in
  rx.io.in <> tx.io.out
  io.out <> rx.io.out
  io.observed      := tx.io.out.bits
  io.observedValid := tx.io.out.valid
}
