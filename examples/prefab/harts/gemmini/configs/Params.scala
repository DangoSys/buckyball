package examples.hart.gemmini.configs

import buckyball.config.HartPlacement
import framework.generic.bus.{axi4, chi => chiBus}

case class Params(chi: chiBus.Params, axi: axi4.Params, resetVector: BigInt) {
  require(axi.addressBits == 44 && axi.dataBits == 128 && axi.idBits == 4)
}

object Params {

  def fromProto(hart: HartPlacement): Params = {
    require(hart.getThreadIndicesCount == 0 && !hart.hasAccelerator && !hart.getService)
    require(hart.getHartId == 0)
    require(
      hart.getImplementation.unpack(classOf[buckyball.components.gemminihart.Params]).getProfile == "DefaultGemmini"
    )
    Params(
      chiBus.Params.fromProto(hart.getMemory.getHome.getRequesters),
      axi4.Params.fromProto(hart.getMemory.getUncached.getAxi),
      BigInt(java.lang.Long.toUnsignedString(hart.getResetVector))
    )
  }

}
