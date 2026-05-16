package framework.system.link

import chisel3.util.log2Ceil
import freechips.rocketchip.amba.axi4.AXI4BundleParameters
import framework.top.GlobalConfig

object BBAxi4Params {

  def apply(b: GlobalConfig): AXI4BundleParameters = {
    AXI4BundleParameters(
      addrBits = b.core.paddrBits,
      dataBits = b.memDomain.dma_buswidth,
      idBits = log2Ceil(b.memDomain.dma_n_xacts).max(1)
    )
  }

}
