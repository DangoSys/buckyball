// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2022 Jiuyang Liu <liu@jiuyang.me>
// Port to buckyball framework

package framework.gpdomain

import chisel3._
import chisel3.util._

package object sequencer {

  /** Find first one */
  def ffo(input: UInt): UInt =
    ((~(scanLeftOr(input) << 1)).asUInt & input)(input.getWidth - 1, 0)

  /** Conditional mask application */
  def maskAnd(mask: Bool, data: Data): Data =
    Mux(mask, data, 0.U.asTypeOf(data))

  /** Enable/disable mask */
  def maskEnable(enable: Bool, mask: UInt): UInt =
    Mux(enable, mask, (-1.S(mask.getWidth.W)).asUInt.asTypeOf(mask))

  /** Convert index to one-hot encoding */
  def indexToOH(index: UInt, chainingSize: Int): UInt =
    UIntToOH(index(log2Ceil(chainingSize), 0))

  /** Check if index matches in one-hot encoded lastReport */
  def ohCheck(lastReport: UInt, index: UInt, chainingSize: Int): Bool =
    (indexToOH(index, chainingSize) & lastReport).orR

  /** Instruction index comparison: a < b */
  def instIndexL(a: UInt, b: UInt): Bool = {
    require(a.getWidth == b.getWidth)
    (a(a.getWidth - 2, 0) < b(b.getWidth - 2, 0)) ^ a(a.getWidth - 1) ^ b(b.getWidth - 1)
  }

  /** Instruction index comparison: a <= b */
  def instIndexLE(a: UInt, b: UInt): Bool = {
    require(a.getWidth == b.getWidth)
    a === b || instIndexL(a, b)
  }

  /** Cut UInt into equal width pieces */
  def cutUInt(data: UInt, width: Int): Vec[UInt] = {
    require(data.getWidth % width == 0)
    VecInit(Seq.tabulate(data.getWidth / width) { groupIndex =>
      data(groupIndex * width + width - 1, groupIndex * width)
    })
  }

  /** Cut UInt into specified number of pieces */
  def cutUIntBySize(data: UInt, size: Int): Vec[UInt] = {
    require(data.getWidth % size == 0)
    val width: Int = data.getWidth / size
    cutUInt(data, width)
  }

  /** Change UInt size with optional sign extension */
  def changeUIntSize(data: UInt, size: Int, sign: Boolean = false): UInt = {
    if (data.getWidth >= size) {
      data(size - 1, 0)
    } else {
      val extend = if (sign) data(data.getWidth - 1) else false.B
      Fill(size - data.getWidth, extend) ## data
    }
  }

  /** Carry-save adder 3:2 compressor */
  def csa32(s: UInt, c: UInt, a: UInt): (UInt, UInt) = {
    val xor = s ^ c
    val so  = xor ^ a
    val co  = (xor & a) | (s & c)
    (so, co)
  }

  /** Multi-lane shifter */
  def multiShifter(right: Boolean, multiSize: Int)(data: UInt, shifterSize: UInt): UInt = {
    VecInit(
      data.asBools
        .grouped(multiSize)
        .toSeq
        .transpose
        .map { dataGroup =>
          if (right) {
            (VecInit(dataGroup).asUInt >> shifterSize).asBools
          } else {
            (VecInit(dataGroup).asUInt << shifterSize).asBools
          }
        }
        .transpose
        .map(VecInit(_).asUInt)
    ).asUInt
  }

}
