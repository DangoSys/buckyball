package framework.system.tile

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import freechips.rocketchip.tilelink._
import framework.system.link.TLXbarModule

/**
 * IOManager - 汇聚所有 Core 的 TileLink 流量
 */
@instantiable
class IOManager(val params: IOManagerParams) extends Module {

  val tlBundleParams = TLBundleParameters(
    addressBits = params.tlAddrBits,
    dataBits = params.tlDataBits,
    sourceBits = params.tlSourceBits,
    sinkBits = 8,
    sizeBits = 4,
    echoFields = Nil,
    requestFields = Nil,
    responseFields = Nil,
    hasBCE = false
  )

  @public
  val io = IO(new Bundle {
    val cores_tl_in = Vec(params.nCores, Flipped(new TLBundle(tlBundleParams)))
    val tl_master   = new TLBundle(tlBundleParams)
  })

  // 使用 Module 版本的 TLXbar
  val xbar = Module(new TLXbarModule(params.nCores, tlBundleParams))
  for (i <- 0 until params.nCores) {
    xbar.io.in(i) <> io.cores_tl_in(i)
  }
  io.tl_master <> xbar.io.out
}
