package examples.goban.tiles.gobantile.configs

import upickle.default._

case class GobanTileConfig(coreConfigs: Seq[String])

object GobanTileConfig {
  implicit val rw: ReadWriter[GobanTileConfig] = macroRW

  def apply(): GobanTileConfig = {
    val jsonStr =
      scala.io.Source.fromFile("src/main/scala/examples/goban/tiles/gobantile/configs/default.json").mkString
    read[GobanTileConfig](jsonStr)
  }

}
