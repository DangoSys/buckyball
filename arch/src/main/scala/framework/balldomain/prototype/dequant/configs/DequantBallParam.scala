package framework.balldomain.prototype.dequant.configs

import upickle.default._

case class DequantBallParam(
  placeholder: Boolean)

object DequantBallParam {
  implicit val rw: ReadWriter[DequantBallParam] = macroRW

  def apply(): DequantBallParam = {
    val jsonStr =
      scala.io.Source.fromFile("src/main/scala/framework/balldomain/prototype/dequant/configs/default.json").mkString
    read[DequantBallParam](jsonStr)
  }

}
