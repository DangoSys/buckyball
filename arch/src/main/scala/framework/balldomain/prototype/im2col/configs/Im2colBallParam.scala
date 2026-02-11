package framework.balldomain.prototype.im2col.configs

import upickle.default._

/**
 * Im2colBall参数
 */
case class Im2colBallParam(
  InputNum:   Int,
  inputWidth: Int)

object Im2colBallParam {
  implicit val rw: ReadWriter[Im2colBallParam] = macroRW

  def apply(): Im2colBallParam = {
    val jsonStr =
      scala.io.Source.fromFile("src/main/scala/framework/balldomain/prototype/im2col/configs/default.json").mkString
    read[Im2colBallParam](jsonStr)
  }

}
