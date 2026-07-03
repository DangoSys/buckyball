package examples.balls.transpose.configs

import upickle.default._

/**
 * TransposeBall Parameter
 */
case class TransposeBallParam(
  InputNum:   Int,
  inputWidth: Int)

object TransposeBallParam {
  implicit val rw: ReadWriter[TransposeBallParam] = macroRW

  def apply(): TransposeBallParam = {
    val jsonStr =
      scala.io.Source.fromFile("../examples/balls/transpose/arch/src/main/scala/configs/default.json").mkString
    read[TransposeBallParam](jsonStr)
  }

}
