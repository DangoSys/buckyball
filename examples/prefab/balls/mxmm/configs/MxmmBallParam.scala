package examples.balls.mxmm.configs

import buckyball.balls.mxmm.{Params => ParametersProto}
import upickle.default.{macroRW, ReadWriter}

case class MxmmBallParam(tileRows: Int, tileCols: Int) {
  require(tileRows == 16 && tileCols == 16)
  val arithmeticLatency = 3
  val contexts          = arithmeticLatency * 2 + 1
}

object MxmmBallParam {

  def fromProto(source: ParametersProto): MxmmBallParam = {
    val p     = source
    require(p.hasTileRows && p.hasTileCols)
    val param = MxmmBallParam(p.getTileRows, p.getTileCols)
    param
  }

  implicit val rw: ReadWriter[MxmmBallParam] = macroRW
}
