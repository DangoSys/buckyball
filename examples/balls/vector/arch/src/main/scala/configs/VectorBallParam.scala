package examples.balls.vector.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig
import upickle.default._

case class VectorBallParam(
  InputNum:      Int,
  inputWidth:    Int,
  lane:          Int,
  outputWidth:   Int,
  numMulThreads: Int,
  numCasThreads: Int
)

object VectorBallParam {
  implicit val rw: ReadWriter[VectorBallParam] = macroRW

  private val ballName = "VecBall"

  def apply(b: GlobalConfig): VectorBallParam = {
    val tbl = BallParamLoader.ballTable(b, ballName)
    VectorBallParam(
      InputNum = BallParamLoader.int(tbl, "InputNum"),
      inputWidth = BallParamLoader.int(tbl, "inputWidth"),
      lane = BallParamLoader.int(tbl, "lane"),
      outputWidth = BallParamLoader.int(tbl, "outputWidth"),
      numMulThreads = BallParamLoader.int(tbl, "numMulThreads"),
      numCasThreads = BallParamLoader.int(tbl, "numCasThreads")
    )
  }
}
