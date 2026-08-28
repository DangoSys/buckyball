package examples.balls.smatmul.configs

import framework.balldomain.configs.BallParamLoader
import framework.top.GlobalConfig

case class SMatMulBallParam(
  InputNum:      Int,
  inputWidth:    Int,
  lane:          Int,
  wsReuseTiles:  Int,
  outputWidth:   Int,
  numMulThreads: Int,
  numCasThreads: Int)

object SMatMulBallParam {
  private val ballName = "SMatMulBall"

  def apply(b: GlobalConfig): SMatMulBallParam = {
    val tbl = BallParamLoader.ballTable(b, ballName)
    SMatMulBallParam(
      InputNum = BallParamLoader.int(tbl, "InputNum"),
      inputWidth = BallParamLoader.int(tbl, "inputWidth"),
      lane = BallParamLoader.int(tbl, "lane"),
      wsReuseTiles = BallParamLoader.int(tbl, "wsReuseTiles"),
      outputWidth = BallParamLoader.int(tbl, "outputWidth"),
      numMulThreads = BallParamLoader.int(tbl, "numMulThreads"),
      numCasThreads = BallParamLoader.int(tbl, "numCasThreads")
    )
  }

}
