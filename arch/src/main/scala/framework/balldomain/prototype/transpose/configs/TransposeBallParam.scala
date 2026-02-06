package framework.balldomain.prototype.transpose.configs

import upickle.default._

/**
 * TransposeBall参数
 */
case class TransposeBallParam(
  InputNum:   Int,
  inputWidth: Int)

object TransposeBallParam {
  implicit val rw: ReadWriter[TransposeBallParam] = macroRW

  def apply(): TransposeBallParam = {
    val jsonStr =
      scala.io.Source.fromFile("src/main/scala/framework/balldomain/prototype/transpose/configs/default.json").mkString
    read[TransposeBallParam](jsonStr)
  }

}
