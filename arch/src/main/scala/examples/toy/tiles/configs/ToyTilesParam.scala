package examples.toy.tiles.configs

import upickle.default._

case class TilesConfig(
  tileConfigs: Seq[String])

object TilesConfig {
  implicit val rw: ReadWriter[TilesConfig] = macroRW

  def apply(): TilesConfig = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/examples/toy/tiles/configs/default.json").mkString
    read[TilesConfig](jsonStr)
  }

}
