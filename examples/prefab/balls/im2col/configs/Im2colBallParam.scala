package examples.balls.im2col.configs

import buckyball.balls.im2col.{Params => ParametersProto}
import upickle.default.{macroRW, ReadWriter}

case class Im2colBallParam(
  maxIter:    Int,
  maxKSize:   Int,
  maxPadding: Int,
  inputWidth: Int)

object Im2colBallParam {

  def fromProto(source: ParametersProto): Im2colBallParam = {
    val parameters = source
    val param      =
      Im2colBallParam(parameters.getMaxIter, parameters.getMaxKSize, parameters.getMaxPadding, parameters.getInputWidth)
    param
  }

  implicit val rw: ReadWriter[Im2colBallParam] = macroRW
}
