package examples.balls.smatmul.configs

import buckyball.balls.smatmul.{Params => ParametersProto}
import upickle.default.{macroRW, ReadWriter}

case class SMatMulBallParam(
  tileRows: Int,
  tileCols: Int)

object SMatMulBallParam {

  def fromProto(source: ParametersProto): SMatMulBallParam = {
    val parameters = source
    require(parameters.hasTileRows && parameters.hasTileCols)
    val param      = SMatMulBallParam(parameters.getTileRows, parameters.getTileCols)
    param
  }

  implicit val rw: ReadWriter[SMatMulBallParam] = macroRW
}
