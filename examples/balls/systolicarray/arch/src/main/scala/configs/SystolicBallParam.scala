package examples.balls.systolicarray.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig

case class SystolicBallParam(
  InputNum:      Int,
  inputWidth:    Int,
  lane:          Int,
  outputWidth:   Int,
  numMulThreads: Int,
  numCasThreads: Int
)

object SystolicBallParam {
  private val ballName = "SystolicArrayBall"

  def apply(b: GlobalConfig): SystolicBallParam = {
    val tbl = BallParamLoader.ballTable(b, ballName)
    SystolicBallParam(
      InputNum = BallParamLoader.int(tbl, "InputNum"),
      inputWidth = BallParamLoader.int(tbl, "inputWidth"),
      lane = BallParamLoader.int(tbl, "lane"),
      outputWidth = BallParamLoader.int(tbl, "outputWidth"),
      numMulThreads = BallParamLoader.int(tbl, "numMulThreads"),
      numCasThreads = BallParamLoader.int(tbl, "numCasThreads")
    )
  }
}
