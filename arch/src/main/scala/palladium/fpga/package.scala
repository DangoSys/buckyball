package object palladium {

  object fpga {
    import org.chipsalliance.cde.config.{Config, Field}

    case object FPGAFrequencyKey extends Field[Double](100.0) // Default to 100MHz
    case object VCU19PTweaksKey  extends Field[Boolean](true)

    class WithFPGAFrequency(freqMHz: Double)
        extends Config((site, here, up) => {
          case FPGAFrequencyKey => freqMHz
        })

    class WithVCU19PTweaks
        extends Config((site, here, up) => {
          case VCU19PTweaksKey => true
        })

  }

}
