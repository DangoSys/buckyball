package examples.toy.balldomain

import chisel3._
import chisel3.util._

// funct7 encoding: [6:4] = enable, [3:0] = opcode
//   enable: 000=no_bank, 001=1rd, 010=1wr, 011=1rd+1wr, 100=2rd+1wr
//   101/110/111 = no_bank (extended space)
object DISA {
  // enable=100 (2 read + 1 write): opcode 0-3
  val MATMUL_WARP16               = BitPat("b1000000") // 64 (0x40)
  val SYSTOLIC                    = BitPat("b1000001") // 65 (0x41)
  val GEMMINI_COMPUTE_PRELOADED   = BitPat("b1000010") // 66 (0x42)
  val GEMMINI_COMPUTE_ACCUMULATED = BitPat("b1000011") // 67 (0x43)

  // enable=011 (1 read + 1 write): opcode 0-6
  val IM2COL          = BitPat("b0110000") // 48 (0x30)
  val TRANSPOSE       = BitPat("b0110001") // 49 (0x31)
  val RELU            = BitPat("b0110010") // 50 (0x32)
  val QUANT           = BitPat("b0110011") // 51 (0x33)
  val DEQUANT         = BitPat("b0110100") // 52 (0x34)
  val GEMMINI_PRELOAD = BitPat("b0110101") // 53 (0x35)
  val BDB_BACKDOOR    = BitPat("b0110110") // 54 (0x36)
  val MXFP            = BitPat("b0110111") // 55 (0x37)

  // enable=000 (no bank access): opcode 2-4
  val GEMMINI_CONFIG = BitPat("b0000010") // 2 (0x02)
  val GEMMINI_FLUSH  = BitPat("b0000011") // 3 (0x03)
  val BDB_COUNTER    = BitPat("b0000100") // 4 (0x04)

  // enable=101 (no bank, extended): Loop WS config, opcode 0-7
  val GEMMINI_LOOP_WS_CONFIG_BOUNDS     = BitPat("b1010000") // 80 (0x50)
  val GEMMINI_LOOP_WS_CONFIG_ADDR_A     = BitPat("b1010001") // 81 (0x51)
  val GEMMINI_LOOP_WS_CONFIG_ADDR_B     = BitPat("b1010010") // 82 (0x52)
  val GEMMINI_LOOP_WS_CONFIG_ADDR_D     = BitPat("b1010011") // 83 (0x53)
  val GEMMINI_LOOP_WS_CONFIG_ADDR_C     = BitPat("b1010100") // 84 (0x54)
  val GEMMINI_LOOP_WS_CONFIG_STRIDES_AB = BitPat("b1010101") // 85 (0x55)
  val GEMMINI_LOOP_WS_CONFIG_STRIDES_DC = BitPat("b1010110") // 86 (0x56)
  val GEMMINI_LOOP_WS                   = BitPat("b1010111") // 87 (0x57)

  // enable=110 (no bank, extended): Loop Conv WS config, opcode 0-9
  val GEMMINI_LOOP_CONV_WS_CONFIG_1 = BitPat("b1100000") // 96  (0x60)
  val GEMMINI_LOOP_CONV_WS_CONFIG_2 = BitPat("b1100001") // 97  (0x61)
  val GEMMINI_LOOP_CONV_WS_CONFIG_3 = BitPat("b1100010") // 98  (0x62)
  val GEMMINI_LOOP_CONV_WS_CONFIG_4 = BitPat("b1100011") // 99  (0x63)
  val GEMMINI_LOOP_CONV_WS_CONFIG_5 = BitPat("b1100100") // 100 (0x64)
  val GEMMINI_LOOP_CONV_WS_CONFIG_6 = BitPat("b1100101") // 101 (0x65)
  val GEMMINI_LOOP_CONV_WS_CONFIG_7 = BitPat("b1100110") // 102 (0x66)
  val GEMMINI_LOOP_CONV_WS_CONFIG_8 = BitPat("b1100111") // 103 (0x67)
  val GEMMINI_LOOP_CONV_WS_CONFIG_9 = BitPat("b1101000") // 104 (0x68)
  val GEMMINI_LOOP_CONV_WS          = BitPat("b1101001") // 105 (0x69)
}
