package examples.poly.tiles.llmtile

import framework.top.GlobalConfig

/**
 * LLM tile — directly delegates to konbi's `KonbiTileConfig`, reusing
 * its heterogeneous prefill+decode core layout. Poly's `llmtile`
 * directory therefore holds no separate descriptors; everything comes
 * from `examples.konbi.tiles.konbitile`.
 */
object LlmTileConfig {

  def apply(): Seq[Option[GlobalConfig]] =
    examples.konbi.tiles.konbitile.KonbiTileConfig()

}
