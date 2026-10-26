package examples.balls.transpose.configs

import buckyball.balls.transpose.{Params => ParametersProto}
import upickle.default.{macroRW, ReadWriter}

case class TransposeBallParam(
  InputNum:   Int,
  inputWidth: Int)

object TransposeBallParam {

  def fromProto(source: ParametersProto): TransposeBallParam = {
    val parameters = source
    require(parameters.hasInputNum && parameters.hasInputWidth)
    val param      = TransposeBallParam(parameters.getInputNum, parameters.getInputWidth)
    param
  }

  implicit val rw: ReadWriter[TransposeBallParam] = macroRW
}
