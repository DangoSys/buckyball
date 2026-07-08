package examples.balls.im2col.configs

import upickle.default._

/**
 * Im2colBall Parameter
 */
case class Im2colBallParam(
  InputNum:   Int,
  inputWidth: Int)

object Im2colBallParam {
  implicit val rw: ReadWriter[Im2colBallParam] = macroRW

  def apply(): Im2colBallParam = {
    val jsonStr =
      scala.io.Source.fromFile("../examples/balls/im2col/arch/src/main/scala/configs/default.json").mkString
    read[Im2colBallParam](jsonStr)
  }

}
