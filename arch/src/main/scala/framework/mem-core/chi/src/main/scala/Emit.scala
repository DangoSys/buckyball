package memcore.bus.chi

object Emit extends App {
  val p = Params()
  require((new RequestFlit(p)).flitWidth == 137)
  require((new ResponseFlit(p)).flitWidth == 71)
  require((new SnoopFlit(p)).flitWidth == 94)
  require((new DataFlit(p)).flitWidth == 389)

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Tx(flitBits = 32, maxCredits = 4),
    firtoolOpts = args,
    args = Array("--target-dir", "build")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Rx(flitBits = 32, depth = 4),
    firtoolOpts = args,
    args = Array("--target-dir", "build")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new SramEndpoint(p),
    firtoolOpts = args,
    args = Array("--target-dir", "build")
  )
}
