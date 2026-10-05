package examples.balls.mxmm

import framework.top.GlobalConfig
import framework.balldomain.configs.{BallDomainParam, BallISAEntry, BallIdMapping}

object Emit extends App {
  require(args.length == 2, "Emit requires OUTPUT_DIRECTORY BALL_CONFIG")
  val defaults = GlobalConfig()

  val b = defaults.copy(
    memDomain = defaults.memDomain.copy(
      bankNum = 8,
      bankWidth = 128,
      bankEntries = 1024,
      bankMaskLen = 16,
      virtualBankCount = 8,
      sharedBankNum = 8,
      mmioBankNum = 8,
      mmioBankEntries = 1024,
      mmioBankWidth = 128,
      mmioReadWidth = 16
    ),
    frontend = defaults.frontend.copy(rob_entries = 16, sub_rob_depth = 8, iter_len = 34),
    ballDomain = BallDomainParam(
      1,
      Seq(BallIdMapping(0, "MxmmBall", "examples.balls.mxmm.MxmmBall", Some(args(1)), 2, 1)),
      Seq(BallISAEntry("MXMM_MXFP8", 71, 0), BallISAEntry("MXMM_FMA32", 72, 0), BallISAEntry("MXMM_F32", 73, 0))
    )
  )

  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new MxmmBall(b),
    args = _root_.scala.Array("--target-dir", args(0), "--split-verilog"),
    firtoolOpts = _root_.scala.Array("--disable-annotation-unknown", "--strip-debug-info")
  )
}
