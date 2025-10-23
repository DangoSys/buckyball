package examples.gemmini.decoder

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.frontend.PostGDCmd
import framework.rocket.RoCCCommandBB

/**
 * GemminiDecoder: 将Gemmini格式的指令转译为Toy格式
 *
 * Gemmini格式:
 * - mvin/mvout: rs1=dram_addr, rs2=(rows<<48)|(cols<<32)|spad_addr, funct=2/3
 *
 * Toy格式:
 * - mvin/mvout: rs1=mem_addr, rs2=(iter<<spAddrLen)|(sp_addr), funct=24/25
 */
class GemminiDecoder(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val gDecoderIn = Flipped(Decoupled(new PostGDCmd))
    val memDecoderOut = Decoupled(new PostGDCmd)
    val ballDecoderOut = Decoupled(new PostGDCmd)
  })

  val func7 = io.gDecoderIn.bits.raw_cmd.inst.funct
  val rs1 = io.gDecoderIn.bits.raw_cmd.rs1
  val rs2 = io.gDecoderIn.bits.raw_cmd.rs2

  // 解析Gemmini的rs2格式
  val gemmini_spad_addr = rs2(31, 0)
  val gemmini_cols = rs2(47, 32)
  val gemmini_rows = rs2(63, 48)

  // 判断是否是Gemmini指令
  val is_gemmini_mvin = func7 === 2.U
  val is_gemmini_mvout = func7 === 3.U
  val is_gemmini_config = func7 === 0.U
  val is_gemmini_compute = (func7 === 4.U) || (func7 === 5.U) // COMPUTE_AND_FLIP or COMPUTE_AND_STAY
  val is_gemmini_preload = func7 === 6.U
  val is_gemmini_flush = func7 === 7.U

  val is_gemmini_mem = is_gemmini_mvin || is_gemmini_mvout  // 只有mvin/mvout走MemDomain
  val is_gemmini_ball = is_gemmini_compute || is_gemmini_preload || is_gemmini_flush || is_gemmini_config  // config走BallDomain

  // 转译为Toy格式
  val toy_mem_addr = rs1  // dram地址直接使用
  val toy_iter = gemmini_rows  // 使用rows作为迭代次数
  val toy_sp_addr = gemmini_spad_addr  // scratchpad地址直接使用

  // 重新组装rs2为Toy格式: (iter<<spAddrLen)|(sp_addr)
  val spAddrLen = b.spAddrLen
  val toy_rs2 = (toy_iter << spAddrLen) | toy_sp_addr

  // 转译后的funct (toy的编码)
  val toy_funct = Mux(is_gemmini_mvin, 24.U,
                  Mux(is_gemmini_mvout, 25.U,
                  func7))  // 其他指令保持不变

  // 创建转译后的命令
  val translated_cmd = Wire(new RoCCCommandBB)
  translated_cmd := io.gDecoderIn.bits.raw_cmd
  translated_cmd.inst.funct := toy_funct
  translated_cmd.rs1 := toy_mem_addr
  translated_cmd.rs2 := toy_rs2

  // 输出到MemDomain (mvin/mvout/config)
  io.memDecoderOut.valid := io.gDecoderIn.valid && is_gemmini_mem
  io.memDecoderOut.bits.is_ball := false.B
  io.memDecoderOut.bits.is_mem := true.B
  io.memDecoderOut.bits.raw_cmd := translated_cmd

  // 输出到BallDomain (compute/preload/flush)
  io.ballDecoderOut.valid := io.gDecoderIn.valid && is_gemmini_ball
  io.ballDecoderOut.bits.is_ball := true.B
  io.ballDecoderOut.bits.is_mem := false.B
  io.ballDecoderOut.bits.raw_cmd := translated_cmd

  // ready信号：根据指令类型选择对应的ready
  io.gDecoderIn.ready := Mux(is_gemmini_mem, io.memDecoderOut.ready,
                         Mux(is_gemmini_ball, io.ballDecoderOut.ready,
                         true.B))  // 未知指令直接丢弃

  // 断言：确保识别的指令是有效的
  assert(!(io.gDecoderIn.fire && !is_gemmini_mem && !is_gemmini_ball),
    "GemminiDecoder: Unknown Gemmini instruction, funct=%d\n", func7)
}
