package examples.balls.fp2int.configs

import upickle.default._

case class Fp2IntBallParam(
  targetType: String // "INT32" or "INT8"
)

object Fp2IntBallParam {
  implicit val rw: ReadWriter[Fp2IntBallParam] = macroRW

  def apply(): Fp2IntBallParam = {
    val jsonStr =
      scala.io.Source.fromFile("../examples/balls/fp2int/arch/src/main/scala/configs/default.json").mkString
    read[Fp2IntBallParam](jsonStr)
  }

}
