package examples.balls.relu.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig
import upickle.default._

case class ReluBallParam(
  InputNum:   Int,
  inputWidth: Int
)

object ReluBallParam {
  implicit val rw: ReadWriter[ReluBallParam] = macroRW

  private val ballName = "ReluBall"

  def apply(b: GlobalConfig): ReluBallParam = {
    val tbl = BallParamLoader.ballTable(b, ballName)
    ReluBallParam(
      InputNum = BallParamLoader.int(tbl, "InputNum"),
      inputWidth = BallParamLoader.int(tbl, "inputWidth")
    )
  }
}
