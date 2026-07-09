package examples.balls.fp2int.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig

case class Fp2IntBallParam(
  targetType: String
)

object Fp2IntBallParam {
  private val ballName = "Fp2IntBall"

  def apply(b: GlobalConfig): Fp2IntBallParam = {
    val tbl = BallParamLoader.ballTable(b, ballName)
    Fp2IntBallParam(targetType = BallParamLoader.str(tbl, "targetType"))
  }
}
