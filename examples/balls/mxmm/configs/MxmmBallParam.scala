package examples.balls.mxmm.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig

case class MxmmBallParam(tileRows: Int, tileCols: Int) {
  require(tileRows == 16 && tileCols == 16)
  val arithmeticLatency = 3
  val contexts          = arithmeticLatency * 2 + 1
}

object MxmmBallParam {

  def apply(b: GlobalConfig): MxmmBallParam = {
    val table = BallParamLoader.ballTable(b, "MxmmBall")
    MxmmBallParam(BallParamLoader.int(table, "tileRows"), BallParamLoader.int(table, "tileCols"))
  }

}
