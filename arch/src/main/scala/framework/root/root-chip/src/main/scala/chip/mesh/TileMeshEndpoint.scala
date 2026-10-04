package hier.chip.mesh

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import memcore.bus.chi._
import memcore.bus.axi.Beat

/**
 * One requester tile's combined coherent and bulk Mesh endpoint.
 *
 * CHI uses VC0--VC3 through `ChiMeshRequesterEndpoint`; bulk AXI streams use
 * VC4. Each VC has independent flow control; physical links arbitrate flits
 * using the receiver credit pool for that VC.
 */
@instantiable
class TileMeshEndpoint(
  p:            Params,
  mesh:         MeshParams,
  localX:       Int,
  localY:       Int,
  nodeMap:      ChiMeshNodeMap,
  bulkDataBits: Int = 256)
    extends Module {
  require(mesh.virtualChannels >= 5)
  require(mesh.payloadBits >= bulkDataBits + bulkDataBits / 8)
  require(nodeMap.coordinateBits == p.nodeIdBits)

  @public
  val io = IO(new Bundle {
    val chi     = Flipped(new RequesterPort(p))
    val bulkIn  = Flipped(Decoupled(new MeshBulkBeat(bulkDataBits, p.nodeIdBits)))
    val bulkOut = Decoupled(new Beat(bulkDataBits))
    val meshOut = Vec(mesh.virtualChannels, Decoupled(new MeshFlit(mesh)))
    val meshIn  = Vec(mesh.virtualChannels, Flipped(Decoupled(new MeshFlit(mesh))))
  })

  val chiEndpoint = Instantiate(new ChiMeshRequesterEndpoint(p, mesh, localX, localY, nodeMap))
  val bulkTx      = Instantiate(new MeshBulkPacketizer(mesh, bulkDataBits, virtualChannel = 4, localX, localY, nodeMap))
  val bulkRx      = Instantiate(new MeshBulkDepacketizer(mesh, bulkDataBits, virtualChannel = 4))
  chiEndpoint.io.chi <> io.chi
  bulkTx.io.in <> io.bulkIn
  for (vc <- 0 until mesh.virtualChannels) {
    if (vc < 4) {
      io.meshOut(vc) <> chiEndpoint.io.meshOut(vc)
      chiEndpoint.io.meshIn(vc) <> io.meshIn(vc)
    } else {
      chiEndpoint.io.meshOut(vc).ready := false.B
      chiEndpoint.io.meshIn(vc).valid  := false.B
      chiEndpoint.io.meshIn(vc).bits   := 0.U.asTypeOf(new MeshFlit(mesh))
      if (vc == 4) { io.meshOut(vc) <> bulkTx.io.out; bulkRx.io.in <> io.meshIn(vc) }
      else {
        io.meshOut(vc).valid := false.B; io.meshOut(vc).bits := 0.U.asTypeOf(new MeshFlit(mesh));
        io.meshIn(vc).ready  := false.B
      }
    }
  }
  io.bulkOut <> bulkRx.io.out
}

/** Native bulk-path verification target for the combined tile endpoint. */
class TileMeshBulkLoopback extends Module {
  private val chi  = Params()
  private val mesh = MeshParams(xNodes = 2, yNodes = 2, payloadBits = 320, virtualChannels = 8)

  private val map = ChiMeshNodeMap(
    Seq.tabulate(1 << chi.nodeIdBits)(_ & 1),
    Seq.tabulate(1 << chi.nodeIdBits)(node => (node >> 1) & 1),
    mesh,
    presentNodes = (0 until (1 << chi.nodeIdBits)).toSet
  )

  @public
  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new MeshBulkBeat(256, chi.nodeIdBits)))
    val out = Decoupled(new Beat(256))
  })

  val endpoint = Instantiate(new TileMeshEndpoint(chi, mesh, 0, 0, map))
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
  endpoint.io.meshIn <> endpoint.io.meshOut
}
