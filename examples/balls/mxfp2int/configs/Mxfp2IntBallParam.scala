package examples.balls.mxfp2int.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig

case class Mxfp2IntBallParam(
  mxfpFormat: String
)

object Mxfp2IntBallParam {
  private val ballName = "Mxfp2IntBall"

  def apply(b: GlobalConfig): Mxfp2IntBallParam = {
    val tbl = BallParamLoader.ballTable(b, ballName)
    Mxfp2IntBallParam(mxfpFormat = BallParamLoader.str(tbl, "mxfpFormat"))
  }
}
