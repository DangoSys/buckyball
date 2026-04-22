package examples.konbi

import org.chipsalliance.cde.config.Config
import examples.konbi.tiles.WithNKonbiTiles

class BuckyballKonbiConfig
    extends Config(
      new WithNKonbiTiles ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )
