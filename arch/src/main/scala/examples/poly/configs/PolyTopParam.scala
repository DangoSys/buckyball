package examples.poly.configs

import upickle.default._

case class PolyConfig(nTiles: Int)

object PolyConfig {
  implicit val rw: ReadWriter[PolyConfig] = macroRW

  def apply(): PolyConfig = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/examples/poly/configs/default.json").mkString
    read[PolyConfig](jsonStr)
  }

}
