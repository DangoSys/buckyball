package examples.poly.tiles.cputile.core.configs

import upickle.default._

/** No fields needed: a CPU-only core has no Buckyball-side parameters. */
case class CpuCoreConfig()

object CpuCoreConfig {
  implicit val rw: ReadWriter[CpuCoreConfig] = macroRW

  def apply(): CpuCoreConfig = {
    val jsonStr =
      scala.io.Source.fromFile("src/main/scala/examples/poly/tiles/cputile/core/configs/default.json").mkString
    read[CpuCoreConfig](jsonStr)
  }

}
