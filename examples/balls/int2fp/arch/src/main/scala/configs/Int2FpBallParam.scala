package examples.balls.int2fp.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig

case class Int2FpBallParam(
  placeholder: Boolean
)

object Int2FpBallParam {
  private val ballName = "Int2FpBall"

  def apply(b: GlobalConfig): Int2FpBallParam = {
    val tbl = BallParamLoader.ballTable(b, ballName)
    Int2FpBallParam(placeholder = BallParamLoader.bool(tbl, "placeholder"))
  }
}
