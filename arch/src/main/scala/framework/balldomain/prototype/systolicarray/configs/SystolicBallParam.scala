package framework.balldomain.prototype.systolicarray.configs

import upickle.default._

/**
 * SystolicBall  Parameter
 */
case class SystolicBallParam(
  InputNum:      Int,
  inputWidth:    Int,
  lane:          Int,
  outputWidth:   Int,
  numMulThreads: Int,
  numCasThreads: Int)

object SystolicBallParam {
  implicit val rw: ReadWriter[SystolicBallParam] = macroRW

  private val configDir =
    "src/main/scala/framework/balldomain/prototype/systolicarray/configs"

  def apply(): SystolicBallParam = fromJson("default")

  def fromJson(name: String): SystolicBallParam = {
    val jsonStr = scala.io.Source.fromFile(s"$configDir/$name.json").mkString
    read[SystolicBallParam](jsonStr)
  }

}
