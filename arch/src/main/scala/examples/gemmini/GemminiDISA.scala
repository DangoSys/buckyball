package examples.gemmini

import chisel3._
import chisel3.util._

/**
 * Gemmini Domain Instruction Set Architecture
 * 定义Gemmini的指令编码
 */
object GemminiDISA {
  // Gemmini的memory指令 (对应原版的LOAD/STORE命令)
  val GEMMINI_CONFIG = BitPat("b0000000")  // CONFIG_CMD = 0
  val GEMMINI_LOAD2  = BitPat("b0000001")  // LOAD2_CMD = 1
  val GEMMINI_LOAD   = BitPat("b0000010")  // LOAD_CMD = 2 (mvin)
  val GEMMINI_STORE  = BitPat("b0000011")  // STORE_CMD = 3 (mvout)

  // Gemmini的compute指令
  val GEMMINI_COMPUTE_AND_FLIP = BitPat("b0000100")  // COMPUTE_AND_FLIP_CMD = 4
  val GEMMINI_COMPUTE_AND_STAY = BitPat("b0000101")  // COMPUTE_AND_STAY_CMD = 5
  val GEMMINI_PRELOAD          = BitPat("b0000110")  // PRELOAD_CMD = 6
  val GEMMINI_FLUSH            = BitPat("b0000111")  // FLUSH_CMD = 7

  val GEMMINI_LOOP_WS                  = BitPat("b0001000")  // LOOP_WS = 8
  val GEMMINI_LOOP_WS_CONFIG_BOUNDS    = BitPat("b0001001")  // 9
  val GEMMINI_LOOP_WS_CONFIG_ADDRS_AB  = BitPat("b0001010")  // 10
  val GEMMINI_LOOP_WS_CONFIG_ADDRS_DC  = BitPat("b0001011")  // 11
  val GEMMINI_LOOP_WS_CONFIG_STRIDES_AB = BitPat("b0001100") // 12
  val GEMMINI_LOOP_WS_CONFIG_STRIDES_DC = BitPat("b0001101") // 13

  val GEMMINI_LOAD3 = BitPat("b0001110")  // LOAD3_CMD = 14

  val GEMMINI_LOOP_CONV_WS          = BitPat("b0001111")  // 15
  val GEMMINI_LOOP_CONV_WS_CONFIG_1 = BitPat("b0010000")  // 16
  val GEMMINI_LOOP_CONV_WS_CONFIG_2 = BitPat("b0010001")  // 17
  val GEMMINI_LOOP_CONV_WS_CONFIG_3 = BitPat("b0010010")  // 18
  val GEMMINI_LOOP_CONV_WS_CONFIG_4 = BitPat("b0010011")  // 19
  val GEMMINI_LOOP_CONV_WS_CONFIG_5 = BitPat("b0010100")  // 20
  val GEMMINI_LOOP_CONV_WS_CONFIG_6 = BitPat("b0010101")  // 21

  val GEMMINI_CLKGATE_EN = BitPat("b0010110")  // 22
}
