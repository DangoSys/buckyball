package memcore.bus.chi

import memcore.bus.chi.snf.SramEndpoint
import memcore.bus.chi.rnf.{BankedChiCache, RnfParams}

object Emit extends App {
  val p = Params()
  require((new RequestFlit(p)).flitWidth == 137)
  require((new ResponseFlit(p)).flitWidth == 71)
  require((new SnoopFlit(p)).flitWidth == 94)
  require((new DataFlit(p)).flitWidth == 389)

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Tx(flitBits = 32, maxCredits = 4),
    firtoolOpts = args,
    args = Array("--target-dir", "build/Tx", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new Rx(flitBits = 32, depth = 4),
    firtoolOpts = args,
    args = Array("--target-dir", "build/Rx", "--split-verilog")
  )
  for ((dataBits, flitBits) <- Seq(128 -> 240, 256 -> 389, 512 -> 687)) {
    val snParams = p.copy(dataBits = dataBits)
    require((new DataFlit(snParams)).flitWidth == flitBits)
    _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
      new SramEndpoint(snParams),
      firtoolOpts = args,
      args = Array("--target-dir", s"build/SramEndpoint$dataBits", "--split-verilog")
    )
  }
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new RequestRetry(p, nodeId = 1),
    firtoolOpts = args,
    args = Array("--target-dir", "build/RequestRetry", "--split-verilog")
  )
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new BankedChiCache(RnfParams()),
    firtoolOpts = args,
    args = Array("--target-dir", "build/BankedChiCache", "--split-verilog")
  )
}
