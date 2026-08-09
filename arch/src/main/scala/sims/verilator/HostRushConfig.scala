package sims.verilator

import org.chipsalliance.cde.config.Config
import framework.system.core.accelerator.BuckyballHostRushKey

/** Selects the DPI host command source for a dedicated Verilator build. */
class WithBuckyballHostRush
    extends Config((site, here, up) => {
      case BuckyballHostRushKey => true
    })
