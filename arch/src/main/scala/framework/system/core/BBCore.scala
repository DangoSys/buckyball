package framework.system.core

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import org.chipsalliance.cde.config.Parameters

import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import freechips.rocketchip.amba.axi4.{AXI4Bundle, AXI4BundleParameters}

import framework.system.sm.SM
import framework.system.link.BBAxi4Params
import framework.system.accelerator.memdomain.frontend.outside_channel.tlb.BBTLBPTWIO

/**
 * BBCore - Core 抽象层
 *
 * 一个 Core = N 个 SM
 *
 * IO 设计:
 * - 每个 SM 独立的 L1 cache 接口 (Vec[FrontendIO], Vec[HellaCacheIO])
 * - 每个 SM 独立的 PTW 接口
 * - 对外: 每个 SM 的 Buckyball DMA AXI4 master (reader + writer)
 *
 * AXI4 → TileLink 协议转换在 BBTile shell 层使用 RocketChip 的 AXI4ToTL 完成
 * (避免自己实现 bridge 的 bug)
 */
@instantiable
class BBCore(val params: BBCoreParams, val coreId: Int)(implicit p: Parameters) extends Module {

  val nSMs         = params.nSMs
  val hasBuckyball = params.smParams.hasBuckyball

  val axiParams: AXI4BundleParameters =
    if (hasBuckyball) BBAxi4Params(params.smParams.buckyballConfig)
    else AXI4BundleParameters(addrBits = 32, dataBits = 64, idBits = 1)

  @public
  val io = IO(new Bundle {
    // === 每个 SM 的控制信号 ===
    val hartids       = Input(Vec(nSMs, UInt(64.W)))
    val reset_vectors = Input(Vec(nSMs, UInt(params.smParams.xLen.W)))
    val interrupts    = Input(Vec(nSMs, new CoreInterrupts(hasBeu = false)))

    // === L1 Cache 接口 (每个 SM 独立) ===
    val imem = Vec(nSMs, new FrontendIO)
    val dmem = Vec(nSMs, new HellaCacheIO)

    // === PTW 接口 (每个 SM 独立) ===
    val ptw_dpath     = Vec(nSMs, Flipped(new DatapathPTWIO))
    val ptw_buckyball = if (hasBuckyball) Some(Vec(nSMs, new BBTLBPTWIO(params.smParams.buckyballConfig))) else None

    // === 对外的 AXI4 master ports (每个 SM 一对 reader + writer)
    // BBTile shell 用 RocketChip 的 AXI4ToTL 转换为 TileLink ===
    val axi_readers = if (hasBuckyball) Some(Vec(nSMs, AXI4Bundle(axiParams))) else None
    val axi_writers = if (hasBuckyball) Some(Vec(nSMs, AXI4Bundle(axiParams))) else None

    // === 状态聚合 ===
    val traces     = Output(Vec(nSMs, new TraceBundle))
    val cease      = Output(Vec(nSMs, Bool()))
    val wfi        = Output(Vec(nSMs, Bool()))
    val traceStall = Input(Bool())
  })

  // === 实例化 N 个 SM ===
  val sms = (0 until nSMs).map { i =>
    val sm = Module(new SM(params.smParams, smId = i))
    sm.io.hartid       := io.hartids(i)
    sm.io.reset_vector := io.reset_vectors(i)
    sm.io.interrupts   := io.interrupts(i)
    sm.io.traceStall   := io.traceStall

    // L1 接口透传
    sm.io.imem <> io.imem(i)
    sm.io.dmem <> io.dmem(i)

    // PTW 透传
    sm.io.ptw_dpath <> io.ptw_dpath(i)
    if (hasBuckyball) {
      sm.io.ptw_buckyball.get <> io.ptw_buckyball.get(i)
    }

    // AXI4 接口透传 (BBTile shell 层做 AXI4ToTL)
    if (hasBuckyball) {
      io.axi_readers.get(i) <> sm.io.buckyball_axi_reader.get
      io.axi_writers.get(i) <> sm.io.buckyball_axi_writer.get
    }

    // 状态聚合
    io.traces(i) := sm.io.trace
    io.cease(i)  := sm.io.cease
    io.wfi(i)    := sm.io.wfi

    sm
  }

}
