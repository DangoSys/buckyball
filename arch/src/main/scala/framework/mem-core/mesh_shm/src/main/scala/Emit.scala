package memcore.memory.mesh_shm

object Emit extends App {
  val p = MeshSharedMemParams.prototype
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new MeshRouter(p, 1, 1),
    firtoolOpts = args,
    args = Array("--target-dir", "build/MeshRouter", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new MeshSharedMem(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build/MeshSharedMem", "--split-verilog")
  )
}
