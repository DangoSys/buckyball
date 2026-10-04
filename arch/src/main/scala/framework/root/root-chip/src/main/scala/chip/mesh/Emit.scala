package hier.chip.mesh

object Emit extends App {
  val p            = MeshParams(xNodes = 3, yNodes = 3, payloadBits = 32, virtualChannels = 4)
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new MeshRouter(p, 1, 1),
    firtoolOpts = args,
    args = Array("--target-dir", "build/MeshRouter", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new MeshCreditNetwork(MeshParams(2, 2, 32, 4), linkDepth = 2),
    firtoolOpts = args,
    args = Array("--target-dir", "build/MeshCreditNetwork", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new ChiMeshCodecLoopback,
    firtoolOpts = args,
    args = Array("--target-dir", "build/ChiMeshCodecLoopback", "--split-verilog")
  )
  val chi          = memcore.bus.chi.Params()
  val endpointMesh = MeshParams(2, 2, 32, 4)

  val endpointNodes = ChiMeshNodeMap(
    Seq.tabulate(1 << chi.nodeIdBits)(_ & 1),
    Seq.tabulate(1 << chi.nodeIdBits)(node => (node >> 1) & 1),
    endpointMesh,
    presentNodes = Set(1, 2, 64)
  )

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new ChiMeshRequesterEndpoint(chi, endpointMesh, 1, 0, endpointNodes),
    firtoolOpts = args,
    args = Array("--target-dir", "build/ChiMeshRequesterEndpoint", "--split-verilog")
  )
}
