package sims.verilator

import org.chipsalliance.cde.config.Config
import framework.system.core.accelerator.BuckyballRushBKey

/** Selects the DPI rushB command source for a dedicated Verilator build. */
class WithBuckyballRushB
    extends Config((site, here, up) => {
      case BuckyballRushBKey => true
    })
