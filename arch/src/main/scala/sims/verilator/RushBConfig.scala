package sims.verilator

import org.chipsalliance.cde.config.Config
import framework.system.core.accelerator.BuckyballRushBKey

/** DPI rushB Verilator build. Broadcast coherence keeps host DRAM staging coherent with DMA. */
class WithBuckyballRushB
    extends Config(
      new chipyard.config.WithBroadcastManager ++
        new Config((site, here, up) => {
          case BuckyballRushBKey => true
        })
    )
