package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig
import framework.balldomain.prototype.gemmini.configs.GemminiBallParam

/**
 * LoopConvAddrGen — combinational im2col address computation.
 *
 * Given loop indices and conv config, computes:
 * - Input DRAM address (with padding detection)
 * - Weight DRAM address
 * - Output DRAM address
 * - Bias DRAM address
 */
@instantiable
class LoopConvAddrGen(val b: GlobalConfig) extends Module {
  val config   = GemminiBallParam()
  val DIM      = config.blockSize
  val elemSize = config.inputWidth / 8
  val accBytes = config.accWidth / 8

  @public
  val io = IO(new Bundle {
    // Conv config
    val cfg        = Input(new LoopConvWsConfig(b))
    // Loop indices
    val batch      = Input(UInt(16.W))
    val orow       = Input(UInt(16.W))
    val ocol       = Input(UInt(16.W))
    val och        = Input(UInt(16.W))
    val krow       = Input(UInt(16.W))
    val kcol       = Input(UInt(16.W))
    val kch        = Input(UInt(16.W))
    // Outputs
    val inputAddr  = Output(UInt(b.memDomain.memAddrLen.W))
    val weightAddr = Output(UInt(b.memDomain.memAddrLen.W))
    val outputAddr = Output(UInt(b.memDomain.memAddrLen.W))
    val biasAddr   = Output(UInt(b.memDomain.memAddrLen.W))
    val isPadding  = Output(Bool())
  })

  val cfg = io.cfg

  // im2col: irow = orow * stride + krow - padding
  val irow = (io.orow * cfg.stride).asSInt + io.krow.asSInt - cfg.padding.asSInt
  val icol = (io.ocol * cfg.stride).asSInt + io.kcol.asSInt - cfg.padding.asSInt

  // Padding detection
  io.isPadding := irow < 0.S || irow >= cfg.in_dim.asSInt ||
    icol < 0.S || icol >= cfg.in_dim.asSInt

  // Input address: base + ((batch * in_dim * in_dim + irow * in_dim + icol) * in_channels + kch) * elemSize
  val inputOffset =
    (io.batch * cfg.in_dim * cfg.in_dim +
      irow.asUInt * cfg.in_dim + icol.asUInt) * cfg.in_channels + io.kch

  io.inputAddr := cfg.dram_addr_input + inputOffset * elemSize.U

  // Weight address: base + ((krow * kernel_dim + kcol) * in_channels + kch) * out_channels * elemSize
  //                      + och * elemSize
  val weightOffset = ((io.krow * cfg.kernel_dim + io.kcol) * cfg.in_channels + io.kch) *
    cfg.out_channels + io.och
  io.weightAddr := cfg.dram_addr_weight + weightOffset * elemSize.U

  // Output address: base + ((batch * out_dim * out_dim + orow * out_dim + ocol) * out_channels + och) * accBytes
  val outputOffset =
    (io.batch * cfg.out_dim * cfg.out_dim +
      io.orow * cfg.out_dim + io.ocol) * cfg.out_channels + io.och

  io.outputAddr := cfg.dram_addr_output + outputOffset * accBytes.U

  // Bias address: base + och * accBytes
  io.biasAddr := cfg.dram_addr_bias + io.och * accBytes.U
}
