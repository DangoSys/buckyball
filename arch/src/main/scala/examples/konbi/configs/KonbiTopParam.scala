package examples.konbi.configs

import upickle.default._

case class KonbiConfig(nTiles: Int)

object KonbiConfig {
  implicit val rw: ReadWriter[KonbiConfig] = macroRW

  def apply(): KonbiConfig = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/examples/konbi/configs/default.json").mkString
    read[KonbiConfig](jsonStr)
  }

}
