package examples.balls.relu.configs

import upickle.default._

/**
 * ReluBall Parameter
 */
case class ReluBallParam(
  InputNum:   Int,
  inputWidth: Int)

object ReluBallParam {
  implicit val rw: ReadWriter[ReluBallParam] = macroRW

  def apply(): ReluBallParam = {
    val jsonStr =
      scala.io.Source.fromFile("../examples/balls/relu/arch/src/main/scala/configs/default.json").mkString
    read[ReluBallParam](jsonStr)
  }

}
