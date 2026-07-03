package examples.balls.int2fp.configs

import upickle.default._

case class Int2FpBallParam(
  placeholder: Boolean)

object Int2FpBallParam {
  implicit val rw: ReadWriter[Int2FpBallParam] = macroRW

  def apply(): Int2FpBallParam = {
    val jsonStr =
      scala.io.Source.fromFile("../examples/balls/int2fp/arch/src/main/scala/configs/default.json").mkString
    read[Int2FpBallParam](jsonStr)
  }

}
