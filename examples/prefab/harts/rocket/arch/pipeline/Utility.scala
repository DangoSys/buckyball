// See LICENSE.SiFive for the Rocket utility contracts.
package freechips.rocketchip

import chisel3._
import chisel3.util._
import scala.language.implicitConversions

package object util {

  implicit class UIntIsOneOf(val x: UInt) extends AnyVal {
    def isOneOf(values: Seq[UInt]): Bool = values.map(x === _).orR
    def isOneOf(first:  UInt, rest: UInt*): Bool = isOneOf(first +: rest.toSeq)
  }

  implicit class SeqBoolBitwiseOps(val x: Seq[Bool]) extends AnyVal {
    def andR:    Bool       = if (x.isEmpty) true.B else x.reduce(_ && _)
    def orR:     Bool       = if (x.isEmpty) false.B else x.reduce(_ || _)
    def xorR:    Bool       = if (x.isEmpty) false.B else x.reduce(_ ^ _)
    def unary_~ : Seq[Bool] = x.map(!_)
    def &(y: Seq[Bool]): Seq[Bool] = (x zip y).map { case (a, b) => a && b }

    def |(y: Seq[Bool]): Seq[Bool] = (x.padTo(y.size, false.B) zip y.padTo(x.size, false.B)).map { case (a, b) =>
      a || b
    }

    def ^(y: Seq[Bool]): Seq[Bool] = (x.padTo(y.size, false.B) zip y.padTo(x.size, false.B)).map { case (a, b) =>
      a ^ b
    }

    def <<(n: Int): Seq[Bool] = Seq.fill(n)(false.B) ++ x
    def >>(n: Int): Seq[Bool] = x.drop(n)
  }

  implicit class BooleanToAugmentedBoolean(val x: Boolean) extends AnyVal {
    def toInt: Int = if (x) 1 else 0
    def option[T](value: => T): Option[T] = if (x) Some(value) else None
  }

  implicit class IntToAugmentedInt(val x: Int) extends AnyVal {
    def log2: Int = { require(isPow2(x)); log2Ceil(x) }
  }

  implicit class UIntToAugmentedUInt(val x: UInt) extends AnyVal {
    def sextTo(n: Int): UInt = { require(x.getWidth <= n); x.asSInt.pad(n).asUInt }
    def padTo(n: Int): UInt = { require(x.getWidth <= n); x.pad(n) }
    def extract(hi: Int, lo: Int): UInt = { require(hi >= lo - 1); if (hi == lo - 1) 0.U else x(hi, lo) }

    def extractOption(hi: Int, lo: Int): Option[UInt] = {
      require(hi >= lo - 1); if (hi == lo - 1) None else Some(x(hi, lo))
    }

    def andNot(y:      UInt): UInt = x & ~(y | (x & 0.U))
    def rotateRight(n: Int):  UInt = if (n == 0) x else Cat(x(n - 1, 0), x >> n)
    def rotateLeft(n:  Int):  UInt = if (n == 0) x else Cat(x(x.getWidth - 1 - n, 0), x(x.getWidth - 1, x.getWidth - n))

    def rotateRight(n: UInt): UInt = (0 until log2Ceil(x.getWidth)).foldLeft(x)((r, i) =>
      Mux(n.padTo(log2Ceil(x.getWidth))(i), r.rotateRight(1 << i), r)
    )

    def rotateLeft(n: UInt): UInt = (0 until log2Ceil(x.getWidth)).foldLeft(x)((r, i) =>
      Mux(n.padTo(log2Ceil(x.getWidth))(i), r.rotateLeft(1 << i), r)
    )

    def grouped(width: Int): Seq[UInt] = (0 until x.getWidth by width).map(i => x(i + width - 1, i))
    def inRange(base:  UInt, bound: UInt): Bool = x >= base && x < bound
    def ##(y:          Option[UInt]): UInt = y.map(x ## _).getOrElse(x)
    def >==(y:         UInt):         Bool = x >= y || y === 0.U
  }

  implicit class OptionUIntToAugmentedOptionUInt(val x: Option[UInt]) extends AnyVal {
    def ##(y: UInt):         UInt         = x.map(_ ## y).getOrElse(y)
    def ##(y: Option[UInt]): Option[UInt] = x.map(_ ## y)
  }

  implicit class DataToAugmentedData[T <: Data](val x: T) {
    def holdUnless(enable: Bool): T = Mux(enable, x, RegEnable(x, enable))
  }

  implicit class VecToAugmentedVec[T <: Data](val x: Vec[T]) {
    def extract(idx: UInt): T = x((idx | 0.U(log2Ceil(x.size).W)).extract(log2Ceil(x.size) - 1, 0))
  }

  implicit class SeqToAugmentedSeq[T <: Data](val x: Seq[T]) {

    def apply(idx: UInt): T =
      if (x.size <= 1) x.head
      else if (!isPow2(x.size))
        (x ++ x.takeRight(x.size & -x.size)).toSeq(idx)
      else {
        val i = (idx | 0.U(log2Ceil(x.size).W))(log2Ceil(x.size) - 1, 0);
        x.zipWithIndex.tail.foldLeft(x.head) { case (r, (v, n)) => Mux(i === n.U, v, r) }
      }

    def extract(idx: UInt): T = VecInit(x).extract(idx)
    def asUInt: UInt = Cat(x.map(_.asUInt).reverse)
  }

  implicit def uintToBitPat(x: UInt):        BitPat = BitPat(x)
  implicit def wcToUInt(x:     WideCounter): UInt   = x.value
  def OH1ToOH(x:               UInt):        UInt   = (x << 1 | 1.U) & ~Cat(0.U(1.W), x)
  def OH1ToUInt(x:             UInt): UInt = OHToUInt(OH1ToOH(x))
  def UIntToOH1(x:             UInt, width: Int): UInt = ~(-1.S(width.W).asUInt << x)(width - 1, 0)
  def UIntToOH1(x:             UInt):        UInt   = UIntToOH1(x, (1 << x.getWidth) - 1)
  def leftOR(x:                UInt):        UInt   = leftOR(x, x.getWidth, x.getWidth)

  def leftOR(x: UInt, width: Int, cap: Int = 999999): UInt = {
    def step(n: Int, y: UInt): UInt = if (n >= width.min(cap)) y else step(n * 2, y | (y << n)(width - 1, 0))
    step(1, x)(width - 1, 0)
  }

  def rightOR(x: UInt): UInt = rightOR(x, x.getWidth, x.getWidth)

  def rightOR(x: UInt, width: Int, cap: Int = 999999): UInt = {
    def step(n: Int, y: UInt): UInt = if (n >= width.min(cap)) y else step(n * 2, y | (y >> n))
    step(1, x)(width - 1, 0)
  }

}
