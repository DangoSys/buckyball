package examples.balls.vector.configs

import buckyball.balls.vector.{Params => ParametersProto}
import upickle.default.{macroRW, ReadWriter}

case class VectorBallParam(
  InputNum:      Int,
  inputWidth:    Int,
  lane:          Int,
  outputWidth:   Int,
  numMulThreads: Int,
  numCasThreads: Int)

object VectorBallParam {

  def fromProto(source: ParametersProto): VectorBallParam = {
    require(
      source.hasInputNum && source.hasInputWidth && source.hasLane && source.hasOutputWidth && source.hasNumMulThreads && source.hasNumCasThreads
    )
    VectorBallParam(
      source.getInputNum,
      source.getInputWidth,
      source.getLane,
      source.getOutputWidth,
      source.getNumMulThreads,
      source.getNumCasThreads
    )
  }

  implicit val rw: ReadWriter[VectorBallParam] = macroRW
}
