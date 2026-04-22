package examples.konbi.tiles.konbitile.configs

import upickle.default._

case class KonbiTileConfig(coreConfigs: Seq[String])

object KonbiTileConfig {
  implicit val rw: ReadWriter[KonbiTileConfig] = macroRW

  def apply(): KonbiTileConfig = {
    val jsonStr =
      scala.io.Source.fromFile("src/main/scala/examples/konbi/tiles/konbitile/configs/default.json").mkString
    read[KonbiTileConfig](jsonStr)
  }

}
