package examples.toy.tiles.toytile.configs

import upickle.default._

case class ToyTileConfig(
  coreConfigs: Seq[String])

object ToyTileConfig {
  implicit val rw: ReadWriter[ToyTileConfig] = macroRW

  def apply(): ToyTileConfig = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/examples/toy/tiles/toytile/configs/default.json").mkString
    read[ToyTileConfig](jsonStr)
  }

}
