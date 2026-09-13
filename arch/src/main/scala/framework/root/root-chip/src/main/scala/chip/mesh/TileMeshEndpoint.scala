package hier.chip.mesh

import chisel3._
import chisel3.util._
import memcore.bus.chi._
import memcore.memory.coherence.ChiRequesterPort
import memcore.bus.axi.AxiSBeat

/**
 * One requester tile's combined coherent and bulk Mesh endpoint.
 *
 * CHI uses VC0--VC3 through `ChiMeshRequesterEndpoint`; bulk AXI streams use
 * VC4. Arbitration happens once per packet, so a bulk transaction retains the
 * physical Mesh port to TLAST after it starts.
 */
class TileMeshEndpoint(
  p:            ChiParams,
  mesh:         MeshParams,
  localX:       Int,
  localY:       Int,
  nodeMap:      ChiMeshNodeMap,
  bulkDataBits: Int = 256)
    extends Module {
  require(mesh.virtualChannels >= 5)
  require(mesh.payloadBits >= bulkDataBits + bulkDataBits / 8)
  require(nodeMap.coordinateBits == p.nodeIdBits)

  val io = IO(new Bundle {
    val chi     = Flipped(new ChiRequesterPort(p))
    val bulkIn  = Flipped(Decoupled(new MeshBulkBeat(bulkDataBits, p.nodeIdBits)))
    val bulkOut = Decoupled(new AxiSBeat(bulkDataBits))
    val meshOut = Decoupled(new MeshFlit(mesh))
    val meshIn  = Flipped(Decoupled(new MeshFlit(mesh)))
  })

  val chiEndpoint = Module(new ChiMeshRequesterEndpoint(p, mesh, localX, localY, nodeMap))
  val bulkTx      = Module(new MeshBulkPacketizer(mesh, bulkDataBits, virtualChannel = 4, localX, localY, nodeMap))
  val bulkRx      = Module(new MeshBulkDepacketizer(mesh, bulkDataBits, virtualChannel = 4))
  val arbiter     = Module(new MeshPacketArbiter(mesh, 2))
  chiEndpoint.io.chi <> io.chi
  bulkTx.io.in <> io.bulkIn
  arbiter.io.in(0) <> chiEndpoint.io.meshOut
  arbiter.io.in(1) <> bulkTx.io.out
  io.meshOut <> arbiter.io.out
  chiEndpoint.io.meshIn.valid := io.meshIn.valid && io.meshIn.bits.vc < 4.U
  chiEndpoint.io.meshIn.bits  := io.meshIn.bits
  bulkRx.io.in.valid          := io.meshIn.valid && io.meshIn.bits.vc === 4.U
  bulkRx.io.in.bits           := io.meshIn.bits
  io.meshIn.ready             := Mux(
    io.meshIn.bits.vc < 4.U,
    chiEndpoint.io.meshIn.ready,
    Mux(io.meshIn.bits.vc === 4.U, bulkRx.io.in.ready, false.B)
  )
  io.bulkOut <> bulkRx.io.out
}

/** Native elaboration target for a tile endpoint sharing coherent and bulk VCs. */
object EmitTileMeshEndpoint extends App {
  private val chi  = ChiParams()
  private val mesh = MeshParams(xNodes = 2, yNodes = 2, payloadBits = 320, virtualChannels = 8)

  private val map = ChiMeshNodeMap(
    Seq.tabulate(1 << chi.nodeIdBits)(_ & 1),
    Seq.tabulate(1 << chi.nodeIdBits)(node => (node >> 1) & 1),
    mesh
  )

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(new TileMeshEndpoint(chi, mesh, 0, 0, map), args)
}

/** Native bulk-path verification target for the combined tile endpoint. */
class TileMeshBulkLoopback extends Module {
  private val chi  = ChiParams()
  private val mesh = MeshParams(xNodes = 2, yNodes = 2, payloadBits = 320, virtualChannels = 8)

  private val map = ChiMeshNodeMap(
    Seq.tabulate(1 << chi.nodeIdBits)(_ & 1),
    Seq.tabulate(1 << chi.nodeIdBits)(node => (node >> 1) & 1),
    mesh
  )

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new MeshBulkBeat(256, chi.nodeIdBits)))
    val out = Decoupled(new AxiSBeat(256))
  })

  val endpoint = Module(new TileMeshEndpoint(chi, mesh, 0, 0, map))
  endpoint.io.bulkIn <> io.in
  io.out <> endpoint.io.bulkOut
  endpoint.io.chi.req.valid   := false.B
  endpoint.io.chi.req.bits    := 0.U.asTypeOf(endpoint.io.chi.req.bits)
  endpoint.io.chi.txRsp.valid := false.B
  endpoint.io.chi.txRsp.bits  := 0.U.asTypeOf(endpoint.io.chi.txRsp.bits)
  endpoint.io.chi.txDat.valid := false.B
  endpoint.io.chi.txDat.bits  := 0.U.asTypeOf(endpoint.io.chi.txDat.bits)
  endpoint.io.chi.snp.ready   := true.B
  endpoint.io.chi.rxRsp.ready := true.B
  endpoint.io.chi.rxDat.ready := true.B
  endpoint.io.meshIn.valid    := endpoint.io.meshOut.valid
  endpoint.io.meshIn.bits     := endpoint.io.meshOut.bits
  endpoint.io.meshOut.ready   := endpoint.io.meshIn.ready
}

object EmitTileMeshBulkLoopback extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(new TileMeshBulkLoopback, args)
}
