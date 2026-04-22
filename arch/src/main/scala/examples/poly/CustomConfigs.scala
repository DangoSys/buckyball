package examples.poly

import org.chipsalliance.cde.config.Config
import examples.poly.tiles.WithNPolyTiles

/**
 * Heterogeneous tile mix:
 *   - one CPU-only tile (cputile)
 *   - one toy-style Buckyball tile (lightdnntile)
 *   - one konbi-style Buckyball tile (llmtile, prefill+decode cores)
 */
class BuckyballPolyConfig
    extends Config(
      new WithNPolyTiles ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )
