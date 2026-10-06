package framework.rvv

import framework.top.GlobalConfig
import framework.balldomain.configs.{BallDomainParam, BallIdMapping}

object Emit extends App {
  val defaults = GlobalConfig()

  val b = defaults.copy(
    rvv = defaults.rvv.copy(enable = true),
    memDomain = defaults.memDomain.copy(
      bankNum = 6,
      bankWidth = 128,
      bankEntries = 256,
      bankMaskLen = 16,
      virtualBankCount = 6,
      nCores = 1,
      computeCoreIds = Seq(0)
    ),
    frontend = defaults.frontend.copy(
      rob_entries = 16,
      bank_id_len = 3,
      vbank_id_upper_bound = 5,
      iter_len = 16
    ),
    ballDomain = BallDomainParam(
      1,
      Seq(BallIdMapping(0, "KernelEngine", "framework.rvv.KernelEngine", None, 4, 4)),
      Seq.empty
    ),
    tile = defaults.tile.copy(xLen = 64)
  )

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new KernelEngine(b),
    firtoolOpts = args,
    args = Array("--target-dir", "build/KernelEngine", "--split-verilog")
  )
}
