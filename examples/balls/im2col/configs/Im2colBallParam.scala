package examples.balls.im2col.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig

case class Im2colBallParam(
  maxIter:    Int,
  maxKSize:   Int,
  maxPadding: Int,
  inputWidth: Int
)

object Im2colBallParam {
  private val ballName = "Im2colBall"

  def apply(b: GlobalConfig): Im2colBallParam = {
    val tbl = BallParamLoader.ballTable(b, ballName)
    Im2colBallParam(
      maxIter = BallParamLoader.int(tbl, "maxIter"),
      maxKSize = BallParamLoader.int(tbl, "maxKSize"),
      maxPadding = BallParamLoader.int(tbl, "maxPadding"),
      inputWidth = BallParamLoader.int(tbl, "inputWidth")
    )
  }
}
