package examples.balls.matrix.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig

case class MatrixBallParam(
  InputNum:      Int,
  inputWidth:    Int,
  lane:          Int,
  outputWidth:   Int,
  numMulThreads: Int,
  numCasThreads: Int
)

object MatrixBallParam {
  private val ballName = "MatrixBall"

  def apply(b: GlobalConfig): MatrixBallParam = {
    val tbl = BallParamLoader.ballTable(b, ballName)
    MatrixBallParam(
      InputNum = BallParamLoader.int(tbl, "InputNum"),
      inputWidth = BallParamLoader.int(tbl, "inputWidth"),
      lane = BallParamLoader.int(tbl, "lane"),
      outputWidth = BallParamLoader.int(tbl, "outputWidth"),
      numMulThreads = BallParamLoader.int(tbl, "numMulThreads"),
      numCasThreads = BallParamLoader.int(tbl, "numCasThreads")
    )
  }
}
