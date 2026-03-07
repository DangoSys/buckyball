package framework.balldomain.prototype.systolicarray.configs

import upickle.default._

/**
 * SystolicBall  Parameter
 */
case class SystolicBallParam(
  InputNum:      Int,
  inputWidth:    Int,
  lane:          Int,
  outputWidth:   Int,
  numMulThreads: Int,
  numCasThreads: Int)

object SystolicBallParam {
  implicit val rw: ReadWriter[SystolicBallParam] = macroRW

  def apply(): SystolicBallParam = {
    val jsonStr = scala.io.Source.fromFile(
      "src/main/scala/framework/balldomain/prototype/systolicarray/configs/default.json"
    ).mkString
    read[SystolicBallParam](jsonStr)
  }

}
