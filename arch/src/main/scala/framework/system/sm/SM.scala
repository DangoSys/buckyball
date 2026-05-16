package framework.system.sm

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import org.chipsalliance.cde.config.Parameters

import freechips.rocketchip.rocket._
import freechips.rocketchip.tile._
import freechips.rocketchip.amba.axi4.AXI4Bundle

import framework.system.sm.rocket.{RocketCore, RocketCoreParams}
import framework.system.accelerator.BuckyballAccelerator
import framework.system.accelerator.memdomain.frontend.outside_channel.tlb.BBTLBPTWIO
import framework.system.link.BBAxi4Params

/**
 * SM (Streaming Multiprocessor)
 *
 * 一个 SM = 一个 Rocket 核心 + 可选的 Buckyball 加速器
 *
 * IO 设计原则:
 * - L1 ICache/DCache 接口暴露出去 (在 Tile shell 层连接到 LazyModule cache)
 * - PTW 接口暴露出去 (每个 SM 独立 PTW)
 * - Buckyball DMA 用 AXI4 (在 Core 层做 AXI4ToTL 桥接)
 */
@instantiable
class SM(val params: SMParams, val smId: Int)(implicit p: Parameters) extends Module {

  val hasBuckyball = params.hasBuckyball

  @public
  val io = IO(new Bundle {
    // === 核心控制 ===
    val hartid       = Input(UInt(64.W))
    val reset_vector = Input(UInt(params.xLen.W))                // 从 SMParams.globalConfig.core.xLen 获取
    val interrupts   = Input(new CoreInterrupts(hasBeu = false)) // hasBeu 由 Core 层管理

    // === L1 Cache 接口 (与 RocketCore 同方向 master, 透传到 Tile shell 的 LazyModule cache) ===
    val imem = new FrontendIO
    val dmem = new HellaCacheIO

    // === PTW 接口 ===
    // RocketCore 内部 ptw 是 Flipped(new DatapathPTWIO), 这里同方向暴露给上层 PTW
    val ptw_dpath     = Flipped(new DatapathPTWIO)
    // Buckyball 的 PTW 接口 (BBTLBPTWIO 包装) - 透传给 Tile shell
    val ptw_buckyball = if (hasBuckyball) Some(new BBTLBPTWIO(params.buckyballConfig)) else None

    // === Buckyball DMA (AXI4) ===
    val buckyball_axi_reader =
      if (hasBuckyball) Some(AXI4Bundle(BBAxi4Params(params.buckyballConfig))) else None
    val buckyball_axi_writer =
      if (hasBuckyball) Some(AXI4Bundle(BBAxi4Params(params.buckyballConfig))) else None

    // === Buckyball 共享后端接口 (透传到 Core 层) ===
    // (后续 Phase 添加 - shared_mem_req, shared_config, barrier 等)

    // === 状态输出 ===
    val trace      = Output(new TraceBundle)
    val cease      = Output(Bool())
    val wfi        = Output(Bool())
    val traceStall = Input(Bool())
  })

  // === 实例化 Rocket 核心 ===
  val rocketParams = RocketCoreParams(
    hasBuckyball = hasBuckyball,
    hasBeu = false,
    nTotalRoCCCSRs = 0
  )

  val rocket = Module(new RocketCore(rocketParams))

  rocket.io.hartid       := io.hartid
  rocket.io.reset_vector := io.reset_vector
  rocket.io.interrupts   := io.interrupts
  rocket.io.imem <> io.imem
  rocket.io.dmem <> io.dmem
  rocket.io.ptw <> io.ptw_dpath
  rocket.io.traceStall   := io.traceStall
  io.trace               := rocket.io.trace
  io.cease               := rocket.io.cease
  io.wfi                 := rocket.io.wfi

  // bpwatch 是 RocketCore 输出, SM 暂不暴露 (上层不需要)
  rocket.io.bpwatch <> DontCare

  // FPU - 暂时 tie off (后续 Phase 接入)
  rocket.io.fpu := DontCare

  // === 实例化 Buckyball (可选) ===
  if (hasBuckyball) {
    val accelerator = Module(new BuckyballAccelerator(params.buckyballConfig))
    accelerator.io.hartid := io.hartid

    // RoCC 接口 (RocketCore.io.rocc 类型为 Flipped(RoCCCoreIOBB))
    accelerator.io.cmd <> rocket.io.rocc.cmd
    rocket.io.rocc.resp <> accelerator.io.resp
    rocket.io.rocc.busy      := accelerator.io.busy
    rocket.io.rocc.interrupt := accelerator.io.interrupt

    // RoCCCoreIOBB.mem / csrs / exception 由 RocketCore 内部使用, accelerator 端不暴露,
    // 这里需要 tie off RocketCore 侧未驱动的输入
    rocket.io.rocc.mem  := DontCare
    rocket.io.rocc.csrs := DontCare

    // AXI4 DMA 接口 (输出到 Core 层)
    io.buckyball_axi_reader.get <> accelerator.io.axi_reader
    io.buckyball_axi_writer.get <> accelerator.io.axi_writer

    // PTW 接口 (输出到 Tile shell 层)
    io.ptw_buckyball.get <> accelerator.io.ptw(0)

    // TLB exception 暂时 tie off (后续 Phase 通过 Core 层连接)
    accelerator.io.tlbExp(0).flush_skip  := false.B
    accelerator.io.tlbExp(0).flush_retry := false.B
    accelerator.io.sfence                := false.B // TODO: 从 Core 层传入

    // 共享后端 - 暂时 tie off (Phase 后续接入 SharedMemBackend)
    // MemRequestIO 方向: write/read 都是 Flipped(SramXXXIO)
    // SramWriteIO/SramReadIO 方向: req = Flipped(Decoupled), resp = Decoupled
    // 两次 Flip 后: req 是 Output (accelerator 发出), resp 是 Input (accelerator 接收)
    accelerator.io.shared_mem_req.foreach { req =>
      // write 通道: req 是 Output (不能驱动), resp 是 Input (需要驱动)
      req.write.req.ready  := false.B // 不接受写请求
      req.write.resp.valid := false.B // 没有写响应
      req.write.resp.bits  := DontCare
      // read 通道: 同上
      req.read.req.ready   := false.B // 不接受读请求
      req.read.resp.valid  := false.B // 没有读响应
      req.read.resp.bits   := DontCare
    }
    accelerator.io.shared_config.ready      := false.B
    accelerator.io.shared_query_group_count := 0.U
    accelerator.io.barrier_release          := false.B
  } else {
    // 没有 Buckyball - tie off RoCC 接口
    rocket.io.rocc.cmd.ready  := false.B
    rocket.io.rocc.resp.valid := false.B
    rocket.io.rocc.resp.bits  := DontCare
    rocket.io.rocc.busy       := false.B
    rocket.io.rocc.interrupt  := false.B
    rocket.io.rocc.mem <> DontCare
    rocket.io.rocc.csrs <> DontCare
  }
}
