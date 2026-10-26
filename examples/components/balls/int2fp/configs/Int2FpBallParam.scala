package examples.balls.int2fp.configs

import buckyball.balls.int2fp.{Params => ParametersProto}
import upickle.default.{macroRW, ReadWriter}

case class Int2FpBallParam(
  inputType:  String,
  outputType: String)

object Int2FpBallParam {

  def fromProto(source: ParametersProto): Int2FpBallParam = {
    val parameters = source
    require(parameters.hasInputType && parameters.hasOutputType)
    val param      = Int2FpBallParam(parameters.getInputType, parameters.getOutputType)
    param
  }

  implicit val rw: ReadWriter[Int2FpBallParam] = macroRW
}
