package examples.goban.configs

import upickle.default._

case class GobanConfig(nTiles: Int)

object GobanConfig {
  implicit val rw: ReadWriter[GobanConfig] = macroRW

  def apply(): GobanConfig = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/examples/goban/configs/default.json").mkString
    read[GobanConfig](jsonStr)
  }

}
