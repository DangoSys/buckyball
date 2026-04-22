package examples.poly.tiles.cputile.configs

import upickle.default._

case class CpuTileConfig(coreConfigs: Seq[String])

object CpuTileConfig {
  implicit val rw: ReadWriter[CpuTileConfig] = macroRW

  def apply(): CpuTileConfig = {
    val jsonStr = scala.io.Source.fromFile("src/main/scala/examples/poly/tiles/cputile/configs/default.json").mkString
    read[CpuTileConfig](jsonStr)
  }

}
