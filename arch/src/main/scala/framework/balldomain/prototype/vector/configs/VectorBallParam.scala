package framework.balldomain.prototype.vector.configs

import upickle.default._

/**
 * VectorBall Parameter
 */
case class VectorBallParam(
  InputNum:      Int,
  inputWidth:    Int,
  lane:          Int,
  outputWidth:   Int,
  numMulThreads: Int,
  numCasThreads: Int)

object VectorBallParam {
  implicit val rw: ReadWriter[VectorBallParam] = macroRW

  def apply(): VectorBallParam = {
    val jsonStr = scala.io.Source.fromFile(
      "src/main/scala/framework/balldomain/prototype/vector/configs/default.json"
    ).mkString
    read[VectorBallParam](jsonStr)
  }

}
