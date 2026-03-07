package framework.balldomain.prototype.quant.configs

import upickle.default._

case class QuantBallParam(
  targetType: String // "INT32" or "INT8"
)

object QuantBallParam {
  implicit val rw: ReadWriter[QuantBallParam] = macroRW

  def apply(): QuantBallParam = {
    val jsonStr =
      scala.io.Source.fromFile("src/main/scala/framework/balldomain/prototype/quant/configs/default.json").mkString
    read[QuantBallParam](jsonStr)
  }

}
