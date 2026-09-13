package examples.balls.smatmul.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig

case class SMatMulBallParam(
  tileRows: Int,
  tileCols: Int)

object SMatMulBallParam {
  private val ballName = "SMatMulBall"

  def apply(b: GlobalConfig): SMatMulBallParam = {
    val tbl = BallParamLoader.ballTable(b, ballName)
    SMatMulBallParam(
      tileRows = BallParamLoader.int(tbl, "tileRows"),
      tileCols = BallParamLoader.int(tbl, "tileCols")
    )
  }

}
