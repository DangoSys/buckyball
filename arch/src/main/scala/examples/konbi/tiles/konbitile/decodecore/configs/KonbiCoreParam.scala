package examples.konbi.tiles.konbitile.decodecore.configs

import upickle.default._

/** Per-core parameters for the decode core variant. */
case class KonbiCoreConfig(balldomain: String)

object KonbiCoreConfig {
  implicit val rw: ReadWriter[KonbiCoreConfig] = macroRW

  def apply(): KonbiCoreConfig = {
    val jsonStr =
      scala.io.Source.fromFile("src/main/scala/examples/konbi/tiles/konbitile/decodecore/configs/default.json").mkString
    read[KonbiCoreConfig](jsonStr)
  }

}
