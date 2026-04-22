package examples.goban.tiles.gobantile.gobancore.configs

import upickle.default._

case class GobanCoreConfig(balldomain: String)

object GobanCoreConfig {
  implicit val rw: ReadWriter[GobanCoreConfig] = macroRW

  def apply(): GobanCoreConfig = {
    val jsonStr =
      scala.io.Source.fromFile("src/main/scala/examples/goban/tiles/gobantile/gobancore/configs/default.json").mkString
    read[GobanCoreConfig](jsonStr)
  }

}
