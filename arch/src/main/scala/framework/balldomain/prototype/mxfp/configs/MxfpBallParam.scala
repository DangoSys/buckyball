package framework.balldomain.prototype.mxfp.configs

import upickle.default._

case class MxfpBallParam(
  InputNum:   Int,
  inputWidth: Int
)

object MxfpBallParam {
  implicit val rw: ReadWriter[MxfpBallParam] = macroRW

  def apply(): MxfpBallParam = {
    val jsonStr =
      scala.io.Source
        .fromFile("src/main/scala/framework/balldomain/prototype/mxfp/configs/default.json")
        .mkString
    read[MxfpBallParam](jsonStr)
  }
}
