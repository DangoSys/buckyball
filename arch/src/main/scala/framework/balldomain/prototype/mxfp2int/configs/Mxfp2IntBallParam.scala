package framework.balldomain.prototype.mxfp2int.configs

import upickle.default._

case class Mxfp2IntBallParam(
  mxfpFormat: String // "MXFP4", "MXFP6", or "MXFP8"
)

object Mxfp2IntBallParam {
  implicit val rw: ReadWriter[Mxfp2IntBallParam] = macroRW

  def apply(): Mxfp2IntBallParam = {
    val jsonStr =
      scala.io.Source
        .fromFile("src/main/scala/framework/balldomain/prototype/mxfp2int/configs/default.json")
        .mkString
    read[Mxfp2IntBallParam](jsonStr)
  }

}
