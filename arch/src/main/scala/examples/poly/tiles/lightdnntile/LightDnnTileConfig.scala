package examples.poly.tiles.lightdnntile

import framework.top.GlobalConfig

/**
 * Light DNN tile — directly delegates to toy's `ToyTileConfig`, reusing
 * its layered JSON layout and Buckyball wiring. Poly's `lightdnntile`
 * directory therefore holds no separate balldomain or core descriptors;
 * everything comes from `examples.toy.tiles.toytile`.
 */
object LightDnnTileConfig {

  def apply(): Seq[Option[GlobalConfig]] =
    examples.toy.tiles.toytile.ToyTileConfig()

}
