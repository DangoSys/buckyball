package examples.balls.mxfp2int.configs

import buckyball.balls.mxfp2int.{Params => ParametersProto}
import upickle.default.{macroRW, ReadWriter}

case class Mxfp2IntBallParam(
  mxfpFormat: String)

object Mxfp2IntBallParam {

  def fromProto(source: ParametersProto): Mxfp2IntBallParam = {
    require(source.hasMxfpFormat)
    Mxfp2IntBallParam(source.getMxfpFormat)
  }

  implicit val rw: ReadWriter[Mxfp2IntBallParam] = macroRW
}
