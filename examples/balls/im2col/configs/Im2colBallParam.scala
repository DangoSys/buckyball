package examples.balls.im2col.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig

case class Im2colBallParam(
  InputNum:   Int,
  inputWidth: Int
)

object Im2colBallParam {
  private val ballName = "Im2colBall"

  def apply(b: GlobalConfig): Im2colBallParam = {
    val tbl = BallParamLoader.ballTable(b, ballName)
    Im2colBallParam(
      InputNum = BallParamLoader.int(tbl, "InputNum"),
      inputWidth = BallParamLoader.int(tbl, "inputWidth")
    )
  }
}
