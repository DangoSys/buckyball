package examples.balls.transpose.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig

case class TransposeBallParam(
  InputNum:   Int,
  inputWidth: Int
)

object TransposeBallParam {
  private val ballName = "TransposeBall"

  def apply(b: GlobalConfig): TransposeBallParam = {
    val tbl = BallParamLoader.ballTable(b, ballName)
    TransposeBallParam(
      InputNum = BallParamLoader.int(tbl, "InputNum"),
      inputWidth = BallParamLoader.int(tbl, "inputWidth")
    )
  }
}
